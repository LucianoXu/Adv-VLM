from typing import Any
import torch
from PIL import Image
from ..image import resized2image01, image012pixel_values

from transformers import AutoTokenizer, AutoModelForMultimodalLM

from .interface import VLM


class LLaVA(VLM):

    IMAGE_TOKEN_ID = 32000
    NUM_IMAGE_TOKENS = 576

    def __init__(self, device: str):
        super().__init__(device)

        self.tok = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.model = AutoModelForMultimodalLM.from_pretrained("llava-hf/llava-1.5-7b-hf")

        # attacks optimize the image, not the weights: freezing params stops backward
        self.model.requires_grad_(False)
        self.model.gradient_checkpointing_enable()
        self.model.eval()

        if self.device is not None:
            self.model.to(self.device)


    def gen(self, img: Image.Image, question: str, answer_priming: str = "", max_new_tokens = 64) -> str:

        text = f"USER: <image>\n{question}\n ASSISTANT:{answer_priming}"
        ids = self.tok(text, return_tensors="pt").input_ids[0]   # BOS is added here

        # expand the placeholders for image tokens
        pos = (ids == self.IMAGE_TOKEN_ID).nonzero()[0, 0].item()
        input_ids = torch.cat([ids[:pos],
                            ids.new_full((self.NUM_IMAGE_TOKENS,), self.IMAGE_TOKEN_ID),
                            ids[pos+1:]])[None].to(self.device)

        pixel_values = image012pixel_values(resized2image01([img])).to(self.device)   # [1, 3, 336, 336]

        generated = input_ids
        past = None                  # KV cache
        cur_input = input_ids        # prefill full sequence, then new tokens only
        pv = pixel_values            # only pass in the image during prefill

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(
                    input_ids=cur_input,
                    attention_mask=torch.ones_like(generated),
                    pixel_values=pv,
                    past_key_values=past,
                    use_cache=True,
                )
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy decoding
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == self.tok.eos_token_id:
                break

            cur_input = next_token
            pv = None

        new_tokens = generated[0][input_ids.shape[-1]:]
        return(self.tok.decode(new_tokens))


    def loglikelyhood_classify(
            self,
            question: str,
            answer_priming: str,
            image01s: torch.Tensor,
            candidates: list[str],
            differentiable: bool = False,
            grad_candidates: list[int] | None = None,
        ) -> torch.Tensor:
        '''
        grad_candidates: Indices of the
        candidates whose scores must carry a gradient back to image01s
        '''

        ctx = torch.enable_grad if differentiable else torch.no_grad

        with ctx():

            img_tensor = image012pixel_values(image01s).to(self.device)

            N = len(img_tensor)

            text = f"USER: <image>\n{question}\n ASSISTANT:{answer_priming}"
            ids = self.tok(text, return_tensors="pt").input_ids[0]   # BOS is added here

            # expand the placeholders for image tokens
            pos = (ids == self.IMAGE_TOKEN_ID).nonzero()[0, 0].item()
            input_ids = torch.cat([ids[:pos],
                                ids.new_full((self.NUM_IMAGE_TOKENS,), self.IMAGE_TOKEN_ID),
                                ids[pos+1:]])[None].to(self.device)

            # N copies for batched inference
            L_prefix = input_ids.shape[1]
            input_ids = input_ids.expand(N, L_prefix)

            if differentiable:
                # Attack path: no KV cache 
                grad_set = set(range(len(candidates))) if grad_candidates is None else set(grad_candidates)
                was_training = self.model.training
                try:
                    scores = []
                    for i, lab in enumerate(candidates):
                        needs_grad = i in grad_set
                        self.model.train(needs_grad)   # checkpointing engages only in train mode; dropouts are 0
                        cand_ctx = torch.enable_grad if needs_grad else torch.no_grad
                        with cand_ctx():
                            lab_ids = self.tok(lab, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                            L_lab = lab_ids.shape[1]
                            lab_ids = lab_ids.expand(N, L_lab)

                            full_ids = torch.cat([input_ids, lab_ids], dim=1)   # (N, L_prefix + L_lab)
                            out = self.model(
                                input_ids=full_ids,
                                attention_mask=torch.ones_like(full_ids),
                                pixel_values=img_tensor,
                                past_key_values=None,
                                use_cache=False,
                            )

                            # logit at position (L_prefix-1 + k) predicts label token k
                            logits = out.logits[:, L_prefix - 1: L_prefix + L_lab - 1, :]            # (N, L_lab, V)
                            lp = logits.log_softmax(-1).gather(2, lab_ids[:, :, None]).squeeze(-1)   # (N, L_lab)
                            scores.append(lp.mean(dim=1))

                    return torch.stack(scores, dim=1)   # (N, X)
                finally:
                    self.model.train(was_training)

            else:
                # precompute the common prefix

                prefix_out = self.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    pixel_values=img_tensor,
                    past_key_values=None,
                    use_cache=True,
                )

                cache = prefix_out.past_key_values
                first_logp = prefix_out.logits[:, -1, :].log_softmax(-1)

                # scores for different labels
                scores = []
                for lab in candidates:
                    lab_ids = self.tok(lab, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                    L_lab = lab_ids.shape[1]
                    lab_ids = lab_ids.expand(N, L_lab)
                    lp0 = first_logp.gather(1, lab_ids[:, :1])  # (N, 1)

                    if L_lab > 1:
                        out = self.model(
                            input_ids = lab_ids,
                            attention_mask = torch.ones(N, L_prefix + L_lab, device=self.device),
                            pixel_values=None,
                            past_key_values=prefix_out.past_key_values,
                            use_cache=True,
                        )


                        rest_logp = out.logits[:, :-1, :].log_softmax(-1).gather(2, lab_ids[:, 1:, None]).squeeze(-1) # (N, L_lab-1)
                        lp = torch.cat([lp0, rest_logp], dim=1)

                        # reset the KV cache (the cache operation is in place)
                        cache.crop(L_prefix)
                    else:
                        lp = lp0

                    scores.append(lp.mean(dim=1))

                return torch.stack(scores, dim=1)   # (N, X)