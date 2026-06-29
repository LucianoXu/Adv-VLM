
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
    def gen(self, img: Image.Image, question: str, answer_priming: str = "", max_new_tokens = 64) -> str:
        '''
        One beam of generation. Return the generated new text.

        img input is in the resized image style.
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