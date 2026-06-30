
from typing import Iterator
from abc import ABC, abstractmethod
from ..model.interface import VLM

import torch


class ImageClass(ABC):

    label_texts: list[str]   # readable label names, in class-index order

    @abstractmethod
    def loader(
        self, batch_size: int, limit: int | None = None, shuffle: bool = False, seed: int = 42
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''
        return image01, labels (torch.Long), indices
        '''
        ...

    @abstractmethod
    def eval_classify_lp(            
        self,
        vlm: VLM,
        question: str,
        answer_priming: str,
        batch_size: int,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> dict:
        ...