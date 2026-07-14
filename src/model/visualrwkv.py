from typing import Literal
from pathlib import Path

import torch
from torch.optim import Adam
from PIL import Image

from ..image import (
    raw2resized, resized2image01, image012pixel_values, quantize_ste,
)
from .interface import VLM
from .visual_rwkv import (
    VisualRWKVModel, build_args, TRIE_TOKENIZER,
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, STOP_TOKEN_INDEX,
)

_VOCAB_FILE = Path(__file__).resolve().parent / "visual_rwkv" / "rwkv_vocab_v20230424.txt"

# RWKV-6 "Finch" 7B language-model backbone.
# grid_size / image_position / ctx_len mirror the 7B finetune (scripts/train/rwkv7b_mix665k.sh):
# grid_size=-1 keeps all 577 CLIP tokens (no pooling); image_position="middle" is the
# "sandwich" prompt the model was trained with.
DEFAULT_ARCH = {
    "n_layer": 32,
    "n_embd": 4096,
    "head_size_a": 64,
    "vocab_size": 65536,
    "grid_size": -1,
    "ctx_len": 2048,
    "vision_tower_name": "openai/clip-vit-large-patch14-336",
    "image_position": "middle",
}


class VisualRWKV(VLM):
    '''
    VisualRWKV-v6 wrapped behind the VLM interface.

    Uses the same CLIP ViT-L/14@336 vision tower as LLaVA, so the
    image01 -> pixel_values pipeline in src/image.py is shared. The RWKV stack
    and its WKV6 CUDA kernel are bfloat16/CUDA only; the model therefore lives in
    bfloat16 on the GPU. Casts to/from the float32 image space stay differentiable,
    so gradient-based attacks still reach the input image.

    Prompt format (single round, image first):
        User: <image>\\n{question}\\n\\nAssistant:{answer_priming}
    '''

    def __init__(self, device: str, model_path: str, arch: dict | None = None):
        super().__init__(device)

        if self.device is None or "cuda" not in str(self.device):
            raise RuntimeError(
                f"VisualRWKV requires a CUDA device (the WKV6 kernel is CUDA-only), got {self.device!r}."
            )

        cfg = {**DEFAULT_ARCH, **(arch or {})}
        self.ctx_len = cfg["ctx_len"]
        self.image_position = cfg["image_position"]
        args = build_args(cfg)

        self.model = VisualRWKVModel(args)
        state = torch.load(model_path, map_location="cpu")
        # transformers >=5 flattened CLIPVisionModel: the checkpoint stores the vision
        # tower under `vit.vision_model.*`, but the current CLIPVisionModel exposes it as
        # `vit.*`. Remap so the (frozen) tower loads from the checkpoint cleanly instead
        # of being silently skipped (it would otherwise fall back to from_pretrained).
        state = {
            (k.replace("vit.vision_model.", "vit.", 1) if k.startswith("vit.vision_model.") else k): v
            for k, v in state.items()
        }
        msg = self.model.load_state_dict(state, strict=False)
        print(" >> VisualRWKV load_state_dict:", msg)
        # rwkv.* and proj.* must load; loud failure beats silent random weights.
        real_missing = [k for k in msg.missing_keys if not k.startswith("vit.")]
        if real_missing:
            raise RuntimeError(f"VisualRWKV checkpoint missing non-vit weights: {real_missing[:8]} ...")

        # attacks optimize the image, not the weights
        self.model.requires_grad_(False)
        self.model.eval()
        self.model = self.model.bfloat16().to(self.device)

        self.tok = TRIE_TOKENIZER(str(_VOCAB_FILE))

    # ------------------------------------------------------------------ helpers

    def _pixel_values(
        self,
        img: Image.Image | torch.Tensor,
        img_type: Literal['raw', 'resized', 'image01', 'pixel_value'],
    ) -> torch.Tensor:
        '''Return CLIP-normalized pixel values [N, 3, 336, 336] (float, on device).'''
        if img_type == 'raw':
            assert isinstance(img, Image.Image)
            return image012pixel_values(resized2image01(raw2resized([img]))).to(self.device)
        elif img_type == 'resized':
            assert isinstance(img, Image.Image)
            return image012pixel_values(resized2image01([img])).to(self.device)
        elif img_type == 'image01':
            assert isinstance(img, torch.Tensor) and img.ndim == 4
            return image012pixel_values(img).to(self.device)
        elif img_type == 'pixel_value':
            assert isinstance(img, torch.Tensor) and img.ndim == 4
            return img.to(self.device)
        else:
            raise ValueError("Invalid image type:", img_type)

    def _prefix_ids(self, question: str, answer_priming: str) -> tuple[list[int], list[int]]:
        '''
        Tokenize the prompt around the single <image> placeholder, matching upstream
        src/dataset.py (process_image_tokens_in_conversations + _add_speaker_and_signal).
        Returns (before_ids, after_ids): token ids before and after the image span.

        image_position (set per the trained checkpoint):
          "middle" -> sandwich:  User: {q}\\n<image>\\n{q}\\n\\nAssistant:{priming}
          "first"  ->            User: <image>\\n{q}\\n\\nAssistant:{priming}
          "last"   ->            User: {q}\\n<image>\\n\\nAssistant:{priming}
        '''
        tail = f"\n\nAssistant:{answer_priming}"
        if self.image_position == "middle":
            prompt = f"User: {question}\n{DEFAULT_IMAGE_TOKEN}\n{question}{tail}"
        elif self.image_position == "first":
            prompt = f"User: {DEFAULT_IMAGE_TOKEN}\n{question}{tail}"
        elif self.image_position == "last":
            prompt = f"User: {question}\n{DEFAULT_IMAGE_TOKEN}{tail}"
        else:
            raise ValueError(f"Unknown image_position: {self.image_position!r}")

        chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
        assert len(chunks) == 2, "prompt must contain exactly one <image> placeholder"
        before_ids = self.tok.encode(chunks[0])
        after_ids = self.tok.encode(chunks[1])
        return before_ids, after_ids

    def _build_prefix(self, pixel_values: torch.Tensor, before_ids: list[int], after_ids: list[int]):
        '''
        Build prefix embeddings [N, L_prefix, n_embd] and the image span indices.

        pixel_values: [N, 3, H, W] (float). image features are computed in bf16.
        Returns (x_prefix, img_start, img_end).
        '''
        N = pixel_values.shape[0]
        images = pixel_values.bfloat16().unsqueeze(1)            # [N, 1, 3, H, W]
        image_features = self.model.encode_images(images)        # [N, n_img, n_embd]
        n_img = image_features.shape[1]

        dev = self.device
        before = self.model.embed_tokens(torch.tensor(before_ids, device=dev)).unsqueeze(0).expand(N, -1, -1)
        after = self.model.embed_tokens(torch.tensor(after_ids, device=dev)).unsqueeze(0).expand(N, -1, -1)

        x_prefix = torch.cat([before, image_features, after], dim=1)   # [N, L_prefix, n_embd]
        img_start = len(before_ids)
        # exclude the trailing CLS token from the bidirectional flip span (upstream
        # preparing_embedding uses image_features.shape[1] - 1); grid_pooling appends CLS last.
        img_end = img_start + n_img - 1
        return x_prefix, img_start, img_end

    # ------------------------------------------------------------------ interface

    def gen(self,
            img: Image.Image | torch.Tensor,
            question: str,
            answer_priming: str = "",
            img_type: Literal['raw', 'resized', 'image01', 'pixel_value'] = 'raw',
            max_new_tokens=64) -> str:

        pixel_values = self._pixel_values(img, img_type)
        assert pixel_values.shape[0] == 1, "gen() handles a single image"

        before_ids, after_ids = self._prefix_ids(question, answer_priming)

        with torch.no_grad():
            x, img_start, img_end = self._build_prefix(pixel_values, before_ids, after_ids)

            generated: list[int] = []
            for _ in range(max_new_tokens):
                logits = self.model.forward_embeds(x, img_start, img_end)[:, -1, :]  # [1, V]
                next_token = int(torch.argmax(logits, dim=-1).item())
                if next_token == STOP_TOKEN_INDEX:
                    break
                generated.append(next_token)
                nxt = self.model.embed_tokens(torch.tensor([[next_token]], device=self.device))
                x = torch.cat([x, nxt], dim=1)
                if x.shape[1] > self.ctx_len:
                    # truncate from the left but keep the image span intact would be ideal;
                    # for short generations this is never hit.
                    x = x[:, -self.ctx_len:, :]

        return self.tok.decode(generated)

    def loglikelyhood_classify(
        self,
        question: str,
        answer_priming: str,
        image01s: torch.Tensor,
        candidates: list[str],
        differentiable: bool = False,
        grad_candidates: list[int] | None = None,
    ) -> torch.Tensor:

        ctx = torch.enable_grad if differentiable else torch.no_grad

        before_ids, after_ids = self._prefix_ids(question, answer_priming)
        
        # Mirror LLaVA: lowercase candidates after a NON-EMPTY answer_priming
        cand_ids = [self.tok.encode(" " + c) for c in candidates]

        with ctx():
            pixel_values = image012pixel_values(image01s).to(self.device)
            N = pixel_values.shape[0]

            x_prefix, img_start, img_end = self._build_prefix(pixel_values, before_ids, after_ids)
            L_prefix = x_prefix.shape[1]

            grad_set = (set(range(len(candidates))) if grad_candidates is None
                        else set(grad_candidates))

            scores = []
            for i, ids in enumerate(cand_ids):
                needs_grad = differentiable and (i in grad_set)
                cand_ctx = torch.enable_grad if needs_grad else torch.no_grad
                with cand_ctx():
                    lab = torch.tensor(ids, device=self.device)
                    L_lab = lab.shape[0]
                    cand_emb = self.model.embed_tokens(lab).unsqueeze(0).expand(N, -1, -1)
                    xx = torch.cat([x_prefix, cand_emb], dim=1)            # [N, L_prefix+L_lab, D]

                    logits = self.model.forward_embeds(xx, img_start, img_end)
                    # logit at position (L_prefix-1 + k) predicts candidate token k
                    logits = logits[:, L_prefix - 1: L_prefix + L_lab - 1, :].float()   # [N, L_lab, V]
                    lab_b = lab[None, :, None].expand(N, L_lab, 1)
                    lp = logits.log_softmax(-1).gather(2, lab_b).squeeze(-1)             # [N, L_lab]
                    scores.append(lp.mean(dim=1))

            return torch.stack(scores, dim=1)   # [N, X]

    def classify_attack(
        self,
        image01s: torch.Tensor,
        question: str,
        answer_priming: str,
        candidates: list[str],
        target_candidate: list[int],
        max_steps: int = 20,
        stop_gap: float = 0.5,
        eps: float = 0.03,
        lr: float = 0.003,
        quantize: bool = False,
    ) -> torch.Tensor:
        '''
        Targeted attack on a whole batch, optimizing each image until its margin
        (target vs. best competitor) exceeds stop_gap, then freezing it. Identical
        in structure to the LLaVA attack -- it only depends on loglikelyhood_classify.
        '''
        proj = quantize_ste if quantize else (lambda x: x)

        N = image01s.shape[0]
        assert len(target_candidate) == N, "need exactly one target per image"

        image01s = image01s.to(self.device)
        target_t = torch.tensor(target_candidate, device=self.device)

        delta = torch.zeros_like(image01s, requires_grad=True)
        opt = Adam([delta], lr=lr)

        done = torch.zeros(N, dtype=torch.bool, device=self.device)
        adv_out = image01s.clone()

        for step in range(max_steps):
            active_idx = (~done).nonzero(as_tuple=False).squeeze(1)
            if active_idx.numel() == 0:
                break

            adv_active = proj(image01s[active_idx] + delta[active_idx])
            tgt_active = target_t[active_idx]
            grad_cols = sorted(set(tgt_active.tolist()))

            scores = self.loglikelyhood_classify(
                question=question,
                answer_priming=answer_priming,
                image01s=adv_active,
                candidates=candidates,
                differentiable=True,
                grad_candidates=grad_cols,
            )

            target_lp = scores.gather(1, tgt_active[:, None]).squeeze(1)   # carries grad

            others = scores.detach().clone()
            others.scatter_(1, tgt_active[:, None], float("-inf"))
            comp_lp = others.max(dim=1).values  # best competitor
            margins = target_lp.detach() - comp_lp

            crossed = margins > stop_gap
            adv_out[active_idx[crossed]] = adv_active.detach()[crossed]
            done[active_idx[crossed]] = True

            print(f"Step {step}  done {int(done.sum())}/{N}  active {active_idx.numel()}  "
                  f"min_margin {margins.min().item():.3f}  mean_margin {margins.mean().item():.3f}")

            if done.all():
                break

            still = ~crossed
            loss = (-target_lp)[still].sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                # hard constraint: project onto the per-example RMS ball, then the [0, 1] box.
                rms = delta.flatten(1).square().mean(dim=1).sqrt()          # (N,)
                delta.mul_((eps / rms.clamp_min(1e-12)).clamp(max=1.0).view(-1, 1, 1, 1))
                delta.copy_((image01s + delta).clamp(0, 1) - image01s)

        rest = (~done).nonzero(as_tuple=False).squeeze(1)
        adv_out[rest] = proj(image01s[rest] + delta[rest]).detach().clamp(0, 1)

        return adv_out


    def saferlhf_attack(
        self,
        path: str,
        init_image: torch.Tensor | None = None,
        batch_size: int = 8,
        limit: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
        max_steps: int = 100,
        lr: float = 0.01,
        quantize: bool = False,
        save_dir: str | None = None,
        save_every: int = 20,
    ) -> torch.Tensor:
        raise NotImplementedError("saferlhf_attack is not implemented for VisualRWKV yet")
