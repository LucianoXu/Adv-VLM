from typing import Literal
from PIL import Image
from abc import ABC, abstractmethod

import torch

from ..utils import resolve_device


class VLM(ABC):
    '''
    Abstract VLM interface. Limit for one round chat with one image.
    '''

    device: str | None

    def __init__(self, device: str | None):
        self.device = resolve_device(device)
    
    @abstractmethod
    def gen(self, 
            img: Image.Image | torch.Tensor, 
            question: str, 
            answer_priming: str = "", 
            img_type: Literal['raw', 'resized', 'image01', 'pixel_value'] = 'raw',
            max_new_tokens = 64) -> str:
        '''
        One beam of generation. Return the generated new text.

        img input should be of the same type as indicated by img_type.
        '''
        ...
 

    @abstractmethod
    def loglikelyhood_classify(
            self,
            question: str,
            answer_priming: str,
            image01s: torch.Tensor,  # shape: (N, C, H, W), values in [0, 1]
            candidates: list[str],
            differentiable: bool = False,
            grad_candidates: list[int] | None = None,
        ) -> torch.Tensor:
        '''
        Evaluate the classification by teacher forcing on the candidates.

        All examples will be processed in one batch.

        If differentiable is True, grad is forced on and gradients flow back to
        image01s (use for gradient-based attacks); otherwise the forward runs under
        torch.no_grad().

        grad_candidates: indices of the candidates
        whose scores carry a gradient back to image01s

        Return a tensor of average log likelyhood. Shape (N, X). N is the number of examples, and X is the number of candidates.
        '''
        ...


    @abstractmethod
    def classify_attack(
        self,
        image01s: torch.Tensor,   # shape: (N, C, H, W), values in [0, 1]
        question: str,
        answer_priming: str,
        candidates: list[str],
        target_candidate: list[int],   # target label index per example, len == N
        max_steps: int = 20,
        stop_gap: float = 0.5,
        eps: float = 0.03,
        lr: float = 0.003,
        quantize: bool = False,
    ) -> torch.Tensor:
        '''
        Targeted adversarial attack on a whole batch of images, run in parallel.

        Each example is optimized until its margin (target vs. best competitor)
        exceeds stop_gap, then frozen. Returns the adversarial image01s, same
        shape as the input, values in [0, 1].

        quantize: if True, attack the resized (uint8) image, otherwise attack continuous
        image01 space.
        '''
        ...


    @abstractmethod
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
        '''
        Universal adversarial attack against the PKU-SafeRLHF dataset.

        Find one image that maximizes the mean teacher-forced log-likelihood of the
        harmful responses across the (prompt, response) pairs loaded from `path`.

        path: path to a PKU-SafeRLHF JSONL file (built into a PKUSafeRLHF internally).

        quantize: if True, attack the resized (uint8) image via a straight-through
                  estimator, otherwise attack continuous image01 space.

        save_dir: if given, the current image is checkpointed into this directory every
                  `save_every` steps (and once more at the end) as both a .pt tensor and
                  a .png. If None, nothing is written.
        save_every: checkpoint interval in steps (ignored when save_dir is None).

        Return the optimized image01, shape (1, C, H, W), values in [0, 1].
        '''
        ...