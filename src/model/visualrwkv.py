from typing import Literal
from pathlib import Path

import torch
from torch.optim import Adam
from PIL import Image

from ..image import (
    raw2resized, resized2image01, image012pixel_values, quantize_ste,
    image012resized, IMAGE_SIZE,
)
from .interface import VLM
from .visual_rwkv import (
    VisualRWKVModel, build_args, TRIE_TOKENIZER,
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, STOP_TOKEN_INDEX,
)
from .visual_rwkv.model import KERNEL_CTXLEN

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

        # single-image / single-prompt case of the stateful batched decoder
        return self.gen_batch(img, [question], answer_priming, img_type, max_new_tokens)[0]

    @torch.no_grad()
    def gen_batch(self,
            img: Image.Image | torch.Tensor,
            questions: list[str],
            answer_priming: str = "",
            img_type: Literal['raw', 'resized', 'image01', 'pixel_value'] = 'raw',
            max_new_tokens: int = 64) -> list[str]:
        '''
        Batched greedy generation. img is either a single image (broadcast) or a batch
        of exactly len(questions) images.

        RWKV is a recurrent stack with no attention mask, and forward_embeds applies a
        single image span to the whole batch, so prompts of different token lengths
        cannot be padded together without corrupting the state / misaligning the image.
        We therefore batch only prompts that share the same (before_len, after_len)
        shape -- those need no padding and share one image span -- and loop across shape
        groups. Identical-length prompts run truly in parallel; fully heterogeneous
        prompts degrade gracefully toward per-prompt decoding, always correct.

        Generated by AI
        '''
        N = len(questions)

        pixel_values = self._pixel_values(img, img_type)      # [M, 3, H, W]
        if pixel_values.shape[0] == 1:
            pixel_values = pixel_values.expand(N, -1, -1, -1)
        assert pixel_values.shape[0] == N, "img must have batch size 1 (broadcast) or len(questions)"

        prefixes = [self._prefix_ids(q, answer_priming) for q in questions]

        # group example indices by (len(before), len(after)) -> padding-free batching
        groups: dict[tuple[int, int], list[int]] = {}
        for i, (before, after) in enumerate(prefixes):
            groups.setdefault((len(before), len(after)), []).append(i)

        results: list[str | None] = [None] * N
        for idxs in groups.values():
            before_batch = [prefixes[i][0] for i in idxs]     # equal length within group
            after_batch = [prefixes[i][1] for i in idxs]
            outs = self._gen_group(pixel_values[idxs], before_batch, after_batch, max_new_tokens)
            for j, i in enumerate(idxs):
                results[i] = outs[j]

        return [r if r is not None else "" for r in results]

    def _gen_group(
        self,
        pixel_values: torch.Tensor,           # [G, 3, H, W]
        before_batch: list[list[int]],        # G id lists, all the same length
        after_batch: list[list[int]],         # G id lists, all the same length
        max_new_tokens: int,
    ) -> list[str]:
        '''
        Parallel greedy decode for a group of equal-shape prompts (no padding).

        Stateful decoding: prefill the prefix once (capturing the RWKV recurrent state),
        then advance one token at a time carrying that state -- O(L) instead of the
        O(L^2) full-sequence recompute per token. Only the causal text suffix is decoded
        incrementally; the bidirectional image scan stays inside the one-shot prefill.
        '''
        G = pixel_values.shape[0]
        dev = self.device

        before = self.model.embed_tokens(torch.tensor(before_batch, device=dev))   # [G, Lb, D]
        after = self.model.embed_tokens(torch.tensor(after_batch, device=dev))     # [G, La, D]
        image_features = self.model.encode_images(pixel_values.bfloat16().unsqueeze(1))  # [G, n_img, D]

        x = torch.cat([before, image_features, after], dim=1)   # [G, L_prefix, D]
        img_start = len(before_batch[0])
        img_end = img_start + image_features.shape[1] - 1       # exclude trailing CLS (see _build_prefix)

        # prefill: last-position logits give the first token; states seed the decode loop
        logits, states = self.model.forward_prefill(x, img_start, img_end)
        nxt = torch.argmax(logits[:, -1, :], dim=-1)                              # [G]

        finished = [False] * G
        gen_tokens: list[list[int]] = [[] for _ in range(G)]
        for _ in range(max_new_tokens):
            for g in range(G):
                if not finished[g]:
                    t = int(nxt[g].item())
                    if t == STOP_TOKEN_INDEX:
                        finished[g] = True
                    else:
                        gen_tokens[g].append(t)
            if all(finished):
                break
            # advance one token (finished rows step too; their output is discarded)
            tok_emb = self.model.embed_tokens(nxt.unsqueeze(1))[:, 0, :]          # [G, D]
            logits, states = self.model.forward_step(tok_emb, states)
            nxt = torch.argmax(logits, dim=-1)                                    # [G]

        return [self.tok.decode(g) for g in gen_tokens]

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


    # ------------------------------------------------------- SafeRLHF universal attack

    @staticmethod
    def _save_image01(image01: torch.Tensor, save_dir: str, tag: str) -> None:
        # checkpoint the current image both as a raw tensor (.pt) and an inspectable image (.png)
        # Byte-for-byte the same helper as LLaVA._save_image01 (same tags / file names), so
        # the two attacks leave directly comparable checkpoint directories behind.
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        img = image01.detach().cpu().clamp(0, 1)
        torch.save(img, out / f"{tag}.pt")
        image012resized(img)[0].save(out / f"{tag}.png")

    def _n_image_tokens(self) -> int:
        '''
        Length of the image span the vision tower contributes (grid_pooling output).

        Measured once with a throw-away forward instead of re-deriving grid_pooling's
        arithmetic here, so it cannot drift from the model when grid_size changes.
        '''
        with torch.no_grad():
            probe = torch.zeros(1, 1, 3, IMAGE_SIZE, IMAGE_SIZE,
                                device=self.device, dtype=torch.bfloat16)
            return int(self.model.encode_images(probe).shape[1])

    def _saferlhf_prepare(
        self,
        prompt: str,
        response: str,
        n_img: int,
        max_response_tokens: int | None,
    ) -> tuple[list[int], list[int], list[int], bool] | None:
        '''
        Tokenize one (prompt, response) pair into (before_ids, after_ids, resp_ids, truncated).
        Returns None when the example cannot be scored at all (empty response, or a prompt
        that on its own overflows the context).

        The prompt goes through _prefix_ids(question=prompt, answer_priming="") so the
        sandwich layout is identical to how VisualRWKV is evaluated everywhere else in this
        codebase. For image_position == "middle" that deliberately repeats the question on
        both sides of the image span -- that is what this checkpoint was trained with.

        The response is encoded with a LEADING SPACE: the prefix ends in "Assistant:" and
        upstream VisualRWKV trains on "Assistant: {response}". This mirrors both
        loglikelyhood_classify (tok.encode(" " + c)) and LLaVA, whose sentencepiece
        tokenizer silently prepends the same space.

        Truncation (a deliberate deviation from LLaVA, which never truncates): the WKV6
        CUDA kernel is compiled with -D_T_=KERNEL_CTXLEN and RUN_CUDA_RWKV6 hard-asserts
        T <= KERNEL_CTXLEN, so one over-long pair would abort the whole run mid-attack.
        SafeRLHF prompts and responses are free-form and "middle" spends the question
        twice, so clipping the response tail beats crashing. It also bounds activation
        memory, which grows linearly in T (see max_response_tokens).
        '''
        if not response.strip():
            # nothing to score. Note the emptiness test is on the raw string, not on the
            # token ids: tok.encode(" ") is a real (space) token, not an empty list.
            return None

        before_ids, after_ids = self._prefix_ids(question=prompt, answer_priming="")
        resp_ids = self.tok.encode(" " + response)

        ctx_cap = min(self.ctx_len, KERNEL_CTXLEN)
        budget = ctx_cap - (len(before_ids) + n_img + len(after_ids))
        if max_response_tokens is not None:
            budget = min(budget, max_response_tokens)
        if budget <= 0:
            return None

        truncated = len(resp_ids) > budget
        if truncated:
            resp_ids = resp_ids[:budget]
        return before_ids, after_ids, resp_ids, truncated

    def _saferlhf_example_loss(
        self,
        image01_single: torch.Tensor,   # (1, 3, H, W), carries grad back to the attacked image
        before_ids: list[int],
        after_ids: list[int],
        resp_ids: list[int],
    ) -> torch.Tensor:
        '''
        Negative MEAN-PER-TOKEN teacher-forced log-likelihood of ONE harmful response,
        conditioned on prompt + the single shared image. Batch dimension is 1.

        This is exactly LLaVA._saferlhf_batch_loss restricted to N == 1: same causal
        shift, same per-example division by the response length. See saferlhf_attack for
        why the batch dimension has to stay at 1 here.
        '''
        pixel_values = image012pixel_values(image01_single).to(self.device)   # (1, 3, H, W)

        x_prefix, img_start, img_end = self._build_prefix(pixel_values, before_ids, after_ids)
        L_prefix = x_prefix.shape[1]

        lab = torch.tensor(resp_ids, device=self.device)
        L_lab = lab.shape[0]
        resp_emb = self.model.embed_tokens(lab).unsqueeze(0)              # (1, L_lab, D)
        xx = torch.cat([x_prefix, resp_emb], dim=1)                       # (1, L_prefix+L_lab, D)

        logits = self.model.forward_embeds(xx, img_start, img_end)
        # causal shift: the logit at position t-1 predicts token t, so this window scores
        # exactly the response tokens (first one predicted by the last prefix position).
        # Slice before .float(): the full [1, T, 65536] logit tensor is bf16, casting only
        # the response window keeps the fp32 copy small.
        logits = logits[:, L_prefix - 1: L_prefix + L_lab - 1, :].float()   # (1, L_lab, V)
        lp = logits.log_softmax(-1).gather(2, lab[None, :, None]).squeeze(-1)   # (1, L_lab)

        # mean over response tokens == LLaVA's logp.sum() / n_response_tokens
        return -lp.mean()

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
        max_response_tokens: int | None = None,
    ) -> torch.Tensor:
        '''
        Universal adversarial attack against PKU-SafeRLHF: find ONE image that maximizes
        the mean teacher-forced log-likelihood of the harmful responses.

        Semantics are identical to LLaVA.saferlhf_attack -- one shared (1, 3, 336, 336)
        image01 optimized directly by Adam (no delta), fully unbounded (the image is only
        clamped back into [0, 1] after each step, there is no eps projection), objective =
        mean over examples of the negative mean-per-token log-likelihood of the response.
        One Adam step is taken per loader batch, cycling the dataset until max_steps steps.

        WHY NO PADDED BATCH: RWKV is a recurrent stack with no attention mask, and
        model.forward_embeds(x, img_start, img_end) applies ONE image span to the whole
        batch. Padding variable-length prompts into a batch would therefore (a) feed pad
        embeddings through the recurrence, corrupting the state of every row that is
        shorter than the longest one -- there is no mask to switch them off -- and (b)
        misalign the single (img_start, img_end) span, which is a per-batch scalar pair,
        against rows whose prompt prefix has a different length. gen_batch works around
        this by grouping prompts of identical (before_len, after_len) shape, but SafeRLHF
        prompts AND responses are all free-form, so no two examples share a shape.
        Instead we interpret `batch_size` as the number of examples ACCUMULATED per
        optimizer step: one forward/backward per example at batch dim 1, no padding at
        all, gradients summed into the same image, then a single Adam step. Because the
        objective is a plain mean over examples, gradient accumulation is EXACT (not an
        approximation) and identical to what a padded batch would have produced. Peak
        activation memory is that of one example, so it does not grow with batch_size.

        max_response_tokens: optional extra cap on the number of scored response tokens
        (LLaVA has no such knob). Responses are already clipped to fit the compiled WKV6
        kernel bound; set this lower if activation memory is tight, since it is linear in
        sequence length.

        If save_dir is given, the current image is checkpointed there every save_every
        steps (and once at the end) as both a .pt tensor and a .png, and per-step training
        metrics (loss, mean log-likelihood, image-gradient norm) are written to a
        TensorBoard event file in the same directory.

        Generated by AI.
        '''
        from ..bench.interface import PKUSafeRLHF   # local import avoids model<->bench cycle

        dataset = PKUSafeRLHF(path, unsafe_only=True)

        # forward through the uint8 grid (attack the resized image) or stay continuous
        proj = quantize_ste if quantize else (lambda x: x)

        if init_image is None:
            g_init = torch.Generator().manual_seed(seed)
            adv = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g_init).to(self.device)
        else:
            adv = init_image.detach().clone()
            if adv.dim() == 3:
                adv = adv[None]
            adv = adv.to(self.device).clamp(0, 1)
        adv.requires_grad_(True)

        opt = Adam([adv], lr=lr)

        # image span length, needed to budget the response against the context cap
        n_img = self._n_image_tokens()

        # TensorBoard: write the training curves into save_dir (skipped when not saving)
        writer = None
        if save_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=save_dir)

        # record the untouched starting image (before any optimizer step)
        if save_dir is not None:
            self._save_image01(proj(adv), save_dir, "adv_init")
            if writer is not None:
                writer.add_image("adv_init", proj(adv)[0].detach().cpu().clamp(0, 1), 0)

        # Unlike LLaVA there is no model.train(True) here: HF needs train mode to engage
        # gradient checkpointing, but this RWKV port drops checkpointing entirely (see the
        # header of visual_rwkv/model.py) and has dropout=0, so eval mode is both correct
        # and identical numerically.
        step = 0
        epoch = 0
        try:
            while step < max_steps:
                for prompts, responses, _is_safe in dataset.loader(
                    batch_size=batch_size, limit=limit, shuffle=shuffle, seed=seed, epoch=epoch
                ):
                    if step >= max_steps:
                        break

                    # tokenize the whole batch first: the divisor of the mean has to be
                    # known before the first backward, since we accumulate example by example
                    prepared = [self._saferlhf_prepare(p, r, n_img, max_response_tokens)
                                for p, r in zip(prompts, responses)]
                    usable = [q for q in prepared if q is not None]
                    n_used = len(usable)
                    n_trunc = sum(int(q[3]) for q in usable)

                    if n_used == 0:
                        print(f"Step {step}  batch 0  SKIPPED "
                              f"(none of the {len(prompts)} examples is scorable)")
                        step += 1
                        continue

                    opt.zero_grad()

                    loss_sum = 0.0
                    for before_ids, after_ids, resp_ids, _trunc in usable:
                        # proj(adv) is rebuilt per example: quantize_ste's tiny graph is
                        # freed by each backward, and re-running it is numerically identical
                        # (forward: snap to the uint8 grid, backward: straight-through).
                        loss_i = self._saferlhf_example_loss(
                            proj(adv), before_ids, after_ids, resp_ids
                        )
                        # weight 1/n_used == the mean over examples LLaVA takes over its batch
                        (loss_i / n_used).backward()
                        loss_sum += loss_i.item()

                    grad_norm = adv.grad.detach().norm().item() if adv.grad is not None else 0.0
                    opt.step()

                    with torch.no_grad():
                        adv.clamp_(0, 1)   # hard constraint: valid image only

                    loss_val = loss_sum / n_used
                    print(f"Step {step}  batch {n_used}  loss {loss_val:.4f}")
                    if n_used != len(prompts) or n_trunc:
                        print(f"    ({len(prompts) - n_used} example(s) skipped, "
                              f"{n_trunc} response(s) truncated to the context cap)")

                    if writer is not None:
                        writer.add_scalar("train/loss", loss_val, step)              # minimized
                        writer.add_scalar("train/mean_logp", -loss_val, step)         # objective (maximized)
                        writer.add_scalar("train/grad_norm", grad_norm, step)
                        writer.add_scalar("train/batch_size", n_used, step)
                        writer.add_scalar("train/n_truncated", n_trunc, step)

                    if save_dir is not None and save_every > 0 and step % save_every == 0:
                        self._save_image01(proj(adv), save_dir, f"adv_step{step:05d}")
                        if writer is not None:
                            writer.add_image("adv_image", proj(adv)[0].detach().cpu().clamp(0, 1), step)

                    step += 1
                epoch += 1
        finally:
            if writer is not None:
                writer.close()

        adv_final = proj(adv).detach().clamp(0, 1)
        if save_dir is not None:
            self._save_image01(adv_final, save_dir, "adv_final")
        return adv_final
