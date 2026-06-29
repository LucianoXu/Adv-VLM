import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from ..envvar import require_hf_token
from ..model.interface import VLM
from ..image import raw2resized, resized2image01
from .interface import ImageClass

# maps Imagenette labels to readable text
SYNSET2NAME = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

class Imagenette(ImageClass):
    HF_REPO = "johnowhitaker/imagenette2-320"

    def __init__(self, local_path: str | None = None):
        '''
        local_path: if given, load the dataset from this on-disk path
        '''
        if local_path is not None:
            self.ds = load_from_disk(local_path)
        else:
            self.HF_TOKEN = require_hf_token()
            self.ds = load_dataset(
                self.HF_REPO,
                split="train", streaming=False,
                token=self.HF_TOKEN
            )
        self.class_synsets = self.ds.features["label"].names
        self.label_texts = [SYNSET2NAME[s] for s in self.class_synsets]

    def loader(self, batch_size: int, limit: int | None = None, shuffle: bool = False, seed: int = 42):
        # return image01, labels (torch.Long), indices

        torch.manual_seed(seed)
        idx = list(range(len(self.ds)))
        if shuffle:
            idx = torch.randperm(len(self.ds)).tolist()   # torch.manual_seed(...) for reproducibility
        if limit is not None:
            idx = idx[:limit]

        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i + batch_size]
            rows = self.ds[batch_idx]   # dict of lists: {'image': [...], 'label': [...]}

            image01 = resized2image01(raw2resized(rows["image"]))
            labels = torch.tensor(rows["label"])
            indices = torch.tensor(batch_idx)
            yield image01, labels, indices

    
    @torch.no_grad()
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

        '''
        Return: a dictionary with the following key-values:
            - question: the question string
            - answer_priming: the answer_priming string
            - batch_size: the batch_size
            - limit: the limit
            - accuracy: the overall accuracy
            - labels: the list of label text (in order)
            - class_accuracy: the list of accuracy for each class, in order
                              (NaN for a class with no evaluated samples)
            - indices: torch.tensor on cpu, dataset indices of the results (processing order)
            - preds: torch.tensor on cpu, the predicted labels of corresponding indices
            - correct_labels: torch.tensor on cpu, the correct labels of corresponding indices
            - logprobs: torch.tensor on cpu, shape (#evaluated, # of labels), the log-likelyhood data

        The three per-example tensors (indices / correct_labels / logprobs) are row-aligned.
        '''

        all_logprobs, all_labels, all_indices = [], [], []

        n_eval = len(self.ds) if limit is None else min(limit, len(self.ds))
        n_batches = (n_eval + batch_size - 1) // batch_size   # ceil

        for image01, labels, indices in tqdm(
            self.loader(batch_size=batch_size, limit=limit, shuffle=shuffle, seed=seed),
            total=n_batches, desc="eval_classify_lp"
        ):

            scores = vlm.loglikelyhood_classify(
                question,
                answer_priming,
                image01.to(vlm.device),
                self.label_texts,
            )   # (B, #labels)

            all_logprobs.append(scores.detach().cpu())
            all_labels.append(labels)
            all_indices.append(indices)

        logprobs = torch.cat(all_logprobs, dim=0)        # (#evaluated, #labels)
        correct_labels = torch.cat(all_labels, dim=0)    # (#evaluated,)
        indices = torch.cat(all_indices, dim=0)          # (#evaluated,)

        preds = logprobs.argmax(dim=1)                   # (#evaluated,)
        correct = preds == correct_labels

        accuracy = correct.float().mean().item()

        class_accuracy = []
        for c in range(len(self.label_texts)):
            mask = correct_labels == c
            class_accuracy.append(correct[mask].float().mean().item() if mask.any() else float("nan"))

        return {
            "question": question,
            "answer_priming": answer_priming,
            "batch_size": batch_size,
            "limit": limit,
            "accuracy": accuracy,
            "labels": self.label_texts,
            "class_accuracy": class_accuracy,
            "indices": indices,
            "preds": preds,
            "correct_labels": correct_labels,
            "logprobs": logprobs,
        }