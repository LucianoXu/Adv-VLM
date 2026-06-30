from typing import Literal
import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from ..envvar import require_hf_token
from ..model.interface import VLM
from ..model.clip import CLIP
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

    @torch.no_grad()
    def eval_classify_clip(
            self,
            clip: CLIP,
            batch_size: int,
            limit: int | None = None,
            shuffle: bool = False,
            seed: int = 42
        ) -> dict:

        '''
        Zero-shot classification with a CLIP encoder, by image-text cosine similarity.

        Return: a dictionary with the following key-values:
            - batch_size: the batch_size
            - limit: the limit
            - accuracy: the overall accuracy
            - labels: the list of label text (in order)
            - class_accuracy: the list of accuracy for each class, in order
                              (NaN for a class with no evaluated samples)
            - indices: torch.tensor on cpu, dataset indices of the results (processing order)
            - preds: torch.tensor on cpu, the predicted labels of corresponding indices
            - correct_labels: torch.tensor on cpu, the correct labels of corresponding indices
            - logits: torch.tensor on cpu, shape (#evaluated, #labels), scaled cosine-sim logits

        The three per-example tensors (indices / correct_labels / logits) are row-aligned.
        '''

        # text features only depend on the label set -> compute once
        text_feat = clip.get_label_feat(self.label_texts)   # (#labels, D)

        all_logits, all_labels, all_indices = [], [], []

        n_eval = len(self.ds) if limit is None else min(limit, len(self.ds))
        n_batches = (n_eval + batch_size - 1) // batch_size   # ceil

        for image01, labels, indices in tqdm(
            self.loader(batch_size=batch_size, limit=limit, shuffle=shuffle, seed=seed),
            total=n_batches, desc="eval_classify_clip"
        ):

            logits = clip.classify(image01, text_feat)   # (B, #labels)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels)
            all_indices.append(indices)

        logits = torch.cat(all_logits, dim=0)            # (#evaluated, #labels)
        correct_labels = torch.cat(all_labels, dim=0)    # (#evaluated,)
        indices = torch.cat(all_indices, dim=0)          # (#evaluated,)

        preds = logits.argmax(dim=1)                     # (#evaluated,)
        correct = preds == correct_labels

        accuracy = correct.float().mean().item()

        class_accuracy = []
        for c in range(len(self.label_texts)):
            mask = correct_labels == c
            class_accuracy.append(correct[mask].float().mean().item() if mask.any() else float("nan"))

        return {
            "batch_size": batch_size,
            "limit": limit,
            "accuracy": accuracy,
            "labels": self.label_texts,
            "class_accuracy": class_accuracy,
            "indices": indices,
            "preds": preds,
            "correct_labels": correct_labels,
            "logits": logits,
        }
    

class ImagenetteAdv(ImageClass):

    def __init__(self, local_path: str, attack_type: Literal['resized', 'image01']):
        '''
        local_path: if given, load the dataset from this on-disk path
        '''
        self.ds = load_from_disk(local_path)
        if attack_type not in ('resized', 'image01'):
            raise ValueError("Invalid attack type:", attack_type)
        self.attack_type = attack_type
        self.label_texts = self.ds.features["original_label"].names


    def loader(self, batch_size: int, limit: int | None = None, shuffle: bool = False, seed: int = 42):
        # return attack_image01, original_labels, attack_labels (torch.Long), indices(in this dataset)

        torch.manual_seed(seed)
        idx = list(range(len(self.ds)))
        if shuffle:
            idx = torch.randperm(len(self.ds)).tolist()   # torch.manual_seed(...) for reproducibility

        if limit is not None:
            idx = idx[:limit]

        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i + batch_size]
            rows = self.ds[batch_idx]   # dict of lists: {'image': [...], 'label': [...]}

            if self.attack_type == 'resized':
                attack_image01 = resized2image01(rows["adversarial_image"])

            elif self.attack_type == 'image01':
                # Array3D float column comes back as nested lists under the default format
                attack_image01 = torch.tensor(rows["adversarial_image"])
            
            else:
                raise Exception()

            original_labels = torch.tensor(rows["original_label"])
            attack_labels = torch.tensor(rows['attack_label'])
            indices = torch.tensor(batch_idx)

            yield attack_image01, original_labels, attack_labels, indices


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

        all_logprobs, all_original_labels, all_attack_labels, all_indices = [], [], [], []

        n_eval = len(self.ds) if limit is None else min(limit, len(self.ds))
        n_batches = (n_eval + batch_size - 1) // batch_size   # ceil

        for attack_image01, original_labels, attack_labels, indices in tqdm(
            self.loader(batch_size=batch_size, limit=limit, shuffle=shuffle, seed=seed),
            total=n_batches, desc="eval_classify_lp"
        ):

            scores = vlm.loglikelyhood_classify(
                question,
                answer_priming,
                attack_image01.to(vlm.device),
                self.label_texts,
            )   # (B, #labels)

            all_logprobs.append(scores.detach().cpu())
            all_original_labels.append(original_labels)
            all_attack_labels.append(attack_labels)
            all_indices.append(indices)

        logprobs = torch.cat(all_logprobs, dim=0)        # (#evaluated, #labels)
        indices = torch.cat(all_indices, dim=0)          # (#evaluated,)

        labels_original = torch.cat(all_original_labels, dim=0)
        labels_attack = torch.cat(all_attack_labels, dim=0)

        preds = logprobs.argmax(dim=1)                   # (#evaluated,)
        correct_original = preds == labels_original
        correct_attack = preds == labels_attack

        accuracy_original = correct_original.float().mean().item()
        accuracy_attack = correct_attack.float().mean().item()

        class_accuracy = []
        for c in range(len(self.label_texts)):
            mask = labels_original == c
            class_accuracy.append(correct_original[mask].float().mean().item() if mask.any() else float("nan"))

        return {
            "question": question,
            "answer_priming": answer_priming,
            "batch_size": batch_size,
            "limit": limit,
            "accuracy_original": accuracy_original,
            "accuracy_attack": accuracy_attack,
            "labels": self.label_texts,
            "class_accuracy": class_accuracy,
            "indices": indices,
            "preds": preds,
            "original_labels": labels_original,
            "attack_labels": labels_attack,
            "logprobs": logprobs,
        }

    @torch.no_grad()
    def eval_classify_clip(
            self,
            clip: CLIP,
            batch_size: int,
            limit: int | None = None,
            shuffle: bool = False,
            seed: int = 42
        ) -> dict:

        '''
        Zero-shot classification of the adversarial images with a CLIP encoder.

        Return: a dictionary with the following key-values:
            - batch_size: the batch_size
            - limit: the limit
            - accuracy_original: how often the prediction still matches the true (original) label
            - accuracy_attack: targeted attack success rate (prediction matches the attack label)
            - labels: the list of label text (in order)
            - class_accuracy: per-class accuracy against the original label, in order
                              (NaN for a class with no evaluated samples)
            - indices: torch.tensor on cpu, dataset indices of the results (processing order)
            - preds: torch.tensor on cpu, the predicted labels of corresponding indices
            - original_labels: torch.tensor on cpu, the true labels of corresponding indices
            - attack_labels: torch.tensor on cpu, the target labels of corresponding indices
            - logits: torch.tensor on cpu, shape (#evaluated, #labels), scaled cosine-sim logits

        The per-example tensors (indices / preds / original_labels / attack_labels / logits) are row-aligned.
        '''

        # text features only depend on the label set -> compute once
        text_feat = clip.get_label_feat(self.label_texts)   # (#labels, D)

        all_logits, all_original_labels, all_attack_labels, all_indices = [], [], [], []

        n_eval = len(self.ds) if limit is None else min(limit, len(self.ds))
        n_batches = (n_eval + batch_size - 1) // batch_size   # ceil

        for attack_image01, original_labels, attack_labels, indices in tqdm(
            self.loader(batch_size=batch_size, limit=limit, shuffle=shuffle, seed=seed),
            total=n_batches, desc="eval_classify_clip"
        ):

            logits = clip.classify(attack_image01, text_feat)   # (B, #labels)

            all_logits.append(logits.detach().cpu())
            all_original_labels.append(original_labels)
            all_attack_labels.append(attack_labels)
            all_indices.append(indices)

        logits = torch.cat(all_logits, dim=0)            # (#evaluated, #labels)
        indices = torch.cat(all_indices, dim=0)          # (#evaluated,)

        labels_original = torch.cat(all_original_labels, dim=0)
        labels_attack = torch.cat(all_attack_labels, dim=0)

        preds = logits.argmax(dim=1)                     # (#evaluated,)
        correct_original = preds == labels_original
        correct_attack = preds == labels_attack

        accuracy_original = correct_original.float().mean().item()
        accuracy_attack = correct_attack.float().mean().item()

        class_accuracy = []
        for c in range(len(self.label_texts)):
            mask = labels_original == c
            class_accuracy.append(correct_original[mask].float().mean().item() if mask.any() else float("nan"))

        return {
            "batch_size": batch_size,
            "limit": limit,
            "accuracy_original": accuracy_original,
            "accuracy_attack": accuracy_attack,
            "labels": self.label_texts,
            "class_accuracy": class_accuracy,
            "indices": indices,
            "preds": preds,
            "original_labels": labels_original,
            "attack_labels": labels_attack,
            "logits": logits,
        }