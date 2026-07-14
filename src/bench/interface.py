
import json
from typing import Iterator
from abc import ABC, abstractmethod
from ..model.interface import VLM
from ..model.clip import CLIP

import torch


class ImageClass(ABC):

    label_texts: list[str]   # readable label names, in class-index order

    @abstractmethod
    def loader(
        self, batch_size: int, limit: int | None = None, shuffle: bool = False, seed: int = 42
    ) -> Iterator[tuple]:
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

    @abstractmethod
    def eval_classify_clip(
        self,
        clip: CLIP,
        batch_size: int,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 42
    ) -> dict:
        ...    


class PKUSafeRLHF:
    
    def __init__(self, path: str, unsafe_only: bool):
        '''
        path: the path for jsonl files in PKU-SafeRLHF signature.
        unsafe_only: if True, keep only the unsafe responses.
        '''
        self.path = path
        self.unsafe_only = unsafe_only

        self.data: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record["prompt"]
                for i in (0, 1):
                    is_safe = record[f"is_response_{i}_safe"]
                    if unsafe_only and is_safe:
                        continue
                    self.data.append({
                        "prompt": prompt,
                        "response": record[f"response_{i}"],
                        "is_safe": is_safe,
                    })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def loader(self, batch_size: int, limit: int | None = None, shuffle: bool = False, seed: int = 42):

        # yield prompts (list[str]), responses (list[str]), is_safe (torch.bool)

        torch.manual_seed(seed)
        idx = list(range(len(self.data)))
        
        if shuffle:
            idx = torch.randperm(len(self.data)).tolist()
        if limit is not None:
            idx = idx[:limit]

        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i + batch_size]
            rows = [self.data[j] for j in batch_idx]

            prompts = [r["prompt"] for r in rows]
            responses = [r["response"] for r in rows]
            is_safe = torch.tensor([r["is_safe"] for r in rows], dtype=torch.bool)
            
            yield prompts, responses, is_safe
