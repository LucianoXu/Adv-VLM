
from PIL import Image
from abc import ABC, abstractmethod

import torch


class VLM(ABC):
    '''
    Abstract VLM interface. Limit for one round chat with one image.
    '''

    device: str

    def __init__(self, device: str):
        self.device = device
    
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
            img: torch.Tensor,  # shape: (N, C, H, W)
            candidates: list[str],
        ) -> torch.Tensor:
        '''
        Evaluate the classification by teacher forcing on the candidates.

        All examples will be processed in one batch.

        Return a tensor of average log likelyhood. Shape (N, X). N is the number of examples, and X is the number of candidates.
        '''
        ...