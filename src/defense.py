"""
Inference-time input transformations as a mitigation baseline.

These are the cheapest credible defence against adversarial images: a fixed
transformation is applied to the image01 tensor just before it enters the
vision encoder, with no retraining of any component.  The point of measuring
them is the dependability trade-off -- how much clean accuracy does a
deployment pay for how much robustness -- not to claim a solved problem.

Every transform maps image01 -> image01: a float tensor (N, 3, H, W) with
values in [0, 1].  Transforms are applied on the batch the model is about to
see, so they cover both the classification and the free-generation paths.

The transforms are deliberately NOT differentiable-friendly (JPEG in
particular is a hard non-differentiable step).  Attacking through them --
BPDA / EOT style adaptive attacks -- is out of scope here, and the wrappers
below refuse to run an attack rather than silently produce a gradient that
does not correspond to the deployed pipeline.
"""

from typing import Callable, Literal
import io

import numpy as np
import torch
from PIL import Image

from .image import IMAGE_SIZE
from .model.interface import VLM


# image01 -> image01
Transform = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------- transforms

def t_identity() -> Transform:
    '''No-op. The control condition, so the defended and undefended paths run
    through exactly the same code.'''
    def apply(image01: torch.Tensor) -> torch.Tensor:
        return image01
    return apply


def t_jpeg(quality: int = 75) -> Transform:
    '''
    JPEG round-trip. Re-encodes each image at the given quality and decodes it
    back, which discards most of the high-frequency content adversarial
    perturbations live in.

    Runs on the CPU through PIL (there is no in-graph JPEG codec); the batch is
    returned on its original device and dtype.
    '''
    def apply(image01: torch.Tensor) -> torch.Tensor:
        device, dtype = image01.device, image01.dtype
        out = []
        for t in image01.detach().cpu().float():
            arr = (t * 255).clamp(0, 255).round().to(torch.uint8)
            im = Image.fromarray(arr.permute(1, 2, 0).numpy())
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            back = np.array(Image.open(buf).convert("RGB"))
            out.append(torch.from_numpy(back).permute(2, 0, 1).float() / 255.0)
        return torch.stack(out, dim=0).to(device=device, dtype=dtype)
    return apply


def t_blur(sigma: float = 1.0) -> Transform:
    '''
    Gaussian blur. Kernel size is derived from sigma as 2*ceil(2*sigma)+1, the
    usual truncation at two standard deviations.
    '''
    from torchvision.transforms.functional import gaussian_blur

    k = 2 * int(np.ceil(2 * sigma)) + 1

    def apply(image01: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(image01, kernel_size=[k, k], sigma=[sigma, sigma]).clamp(0, 1)
    return apply


def t_bit_depth(bits: int = 3) -> Transform:
    '''
    Bit-depth reduction: snap each channel to 2**bits evenly spaced levels.
    At bits=8 this is the identity on any image that came off a uint8 file.
    '''
    levels = 2 ** bits - 1

    def apply(image01: torch.Tensor) -> torch.Tensor:
        return (image01.clamp(0, 1) * levels).round() / levels
    return apply


def t_resize_pad(scale_min: float = 0.85, scale_max: float = 1.0, seed: int = 42) -> Transform:
    '''
    Random resize-and-pad: shrink the image by a random factor, then place it at
    a random offset on a zero canvas of the original size. Breaks the pixel
    alignment an adversarial perturbation was optimized for.

    The randomness is drawn from a transform-local seeded generator, so a whole
    evaluation is reproducible while still varying across images.
    '''
    import torch.nn.functional as F

    gen = torch.Generator().manual_seed(seed)

    def apply(image01: torch.Tensor) -> torch.Tensor:
        N, C, H, W = image01.shape
        out = torch.zeros_like(image01)
        for i in range(N):
            s = scale_min + (scale_max - scale_min) * torch.rand(1, generator=gen).item()
            h, w = max(1, int(round(H * s))), max(1, int(round(W * s)))
            small = F.interpolate(
                image01[i: i + 1], size=(h, w), mode="bilinear", align_corners=False
            )
            top = int(torch.randint(0, H - h + 1, (1,), generator=gen).item())
            left = int(torch.randint(0, W - w + 1, (1,), generator=gen).item())
            out[i: i + 1, :, top: top + h, left: left + w] = small
        return out.clamp(0, 1)
    return apply


TRANSFORMS = {
    "identity": t_identity,
    "jpeg": t_jpeg,
    "blur": t_blur,
    "bit_depth": t_bit_depth,
    "resize_pad": t_resize_pad,
}


def transform_factory(defense_args: dict) -> Transform:
    '''
    Build a transform from a config block, e.g.

        defense:
          name: jpeg
          quality: 75

    Any key other than `name` is passed to the transform constructor.
    '''
    args = dict(defense_args)
    name = args.pop("name")
    if name not in TRANSFORMS:
        raise ValueError(f"unknown defense {name!r}; known: {sorted(TRANSFORMS)}")
    print(f" >> Defense: {name} {args}")
    return TRANSFORMS[name](**args)


# ------------------------------------------------------------------ wrappers

class DefendedVLM(VLM):
    '''
    A VLM with an input transformation spliced in front of its vision path.

    Delegates every method to the wrapped model, transforming image01 inputs on
    the way through. Only `img_type='image01'` is transformed -- that is the
    format every evaluation in this project feeds -- and any other img_type is
    rejected rather than passed through undefended.
    '''

    def __init__(self, inner: VLM, transform: Transform):
        self.inner = inner
        self.transform = transform
        self.device = inner.device

    def _guard(self, img_type: str) -> None:
        if img_type != "image01":
            raise ValueError(
                f"DefendedVLM only accepts img_type='image01', got {img_type!r}. "
                "Convert to image01 first, otherwise the defence would be skipped silently."
            )

    def gen(self, img, question, answer_priming="", img_type="raw", max_new_tokens=64) -> str:
        self._guard(img_type)
        return self.inner.gen(
            self.transform(img), question, answer_priming, img_type, max_new_tokens
        )

    def gen_batch(self, img, questions, answer_priming="", img_type="raw", max_new_tokens=64):
        self._guard(img_type)
        return self.inner.gen_batch(
            self.transform(img), questions, answer_priming, img_type, max_new_tokens
        )

    def loglikelyhood_classify(
        self, question, answer_priming, image01s, candidates,
        differentiable=False, grad_candidates=None,
    ) -> torch.Tensor:
        if differentiable:
            raise NotImplementedError(
                "differentiable=True through a DefendedVLM would attack the wrong "
                "pipeline: these transforms are non-differentiable by design. "
                "Adaptive (BPDA/EOT) attacks are out of scope."
            )
        return self.inner.loglikelyhood_classify(
            question, answer_priming, self.transform(image01s), candidates,
            differentiable=False, grad_candidates=grad_candidates,
        )

    def classify_attack(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "attacking a DefendedVLM is not supported; craft the attack on the "
            "undefended model and evaluate it through the defence."
        )

    def saferlhf_attack(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "attacking a DefendedVLM is not supported; craft the attack on the "
            "undefended model and evaluate it through the defence."
        )


class DefendedCLIP:
    '''
    The CLIP counterpart of DefendedVLM. CLIP is not a VLM subclass, so this is
    a plain delegating wrapper exposing the two methods the evaluation uses.
    '''

    def __init__(self, inner, transform: Transform):
        self.inner = inner
        self.transform = transform
        self.device = inner.device

    def get_label_feat(self, labels, template: str = "a photo of a {}") -> torch.Tensor:
        return self.inner.get_label_feat(labels, template)

    def classify(self, image01s: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        return self.inner.classify(self.transform(image01s), text_feat)

    def attack(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "attacking a DefendedCLIP is not supported; craft the attack on the "
            "undefended encoder and evaluate it through the defence."
        )


def apply_defense(model, defense_args: dict | None):
    '''
    Wrap `model` if a defense block is configured, otherwise return it as-is.
    Dispatches on whether the model is a VLM or the CLIP encoder.
    '''
    if not defense_args:
        return model
    transform = transform_factory(defense_args)
    if isinstance(model, VLM):
        return DefendedVLM(model, transform)
    return DefendedCLIP(model, transform)
