import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import CLIPModel, CLIPTokenizerFast

from src.utils import resolve_device
from src.image import image012pixel_values, quantize_ste


class CLIP:

    CLIP_ID = "openai/clip-vit-large-patch14-336"

    def __init__(self, device: str | None):
        self.device = resolve_device(device)
        self.clip = CLIPModel.from_pretrained(self.CLIP_ID).eval()

        if self.device is not None:
            self.clip.to(self.device)

        self.clip.requires_grad_(False)
        self.clip_tok = CLIPTokenizerFast.from_pretrained(self.CLIP_ID)

    def get_label_feat(
        self,
        labels: list[str],
        template: str = "a photo of a {}",
    ) -> torch.Tensor:
        
        prompts = [template.format(name) for name in labels]

        with torch.no_grad():
            tok = self.clip_tok(prompts, padding=True, return_tensors="pt").to(self.device)
            text_feat = self.clip.get_text_features(**tok).pooler_output
            text_feat = F.normalize(text_feat, dim=-1)

        return text_feat


    def classify(
        self,
        image01s: torch.Tensor,
        text_feat: torch.Tensor,    # (#labels, 768)
    ) -> torch.Tensor:
        '''
        image01: (N, 3, 336, 336)
        return logits (N, #labels)
        '''
        with torch.no_grad():
            pixel_values = image012pixel_values(image01s).to(self.device)
            
            img_feat = self.clip.get_image_features(pixel_values=pixel_values).pooler_output
            img_feat = F.normalize(img_feat, dim=-1)
            logit_scale = self.clip.logit_scale.exp()
            return logit_scale * img_feat @ text_feat.t()
        

    def attack(
        self,
        image01s: torch.Tensor,
        target_texts: list[str],
        label_template: str = "a photo of a {}",
        
        train_steps: int = 20,
        gamma: float = 1.0,
        lr: float = 0.003,
        quantize: bool = False,
    ) -> torch.Tensor:

        '''
        Train a fixed number of steps to fit the image01s towards the target texts by cosine similarity.
        '''
        # forward through the uint8 grid or stay continuous
        proj = quantize_ste if quantize else (lambda x: x)

        N = image01s.shape[0]
        assert len(target_texts) == N, "need exactly one target per image"

        image01s = image01s.to(self.device)
        delta = torch.zeros_like(image01s, requires_grad=True).to(self.device)

        opt = Adam(
            [delta],
            lr = lr
        )

        # get the features for ones
        target_feats = self.get_label_feat(target_texts, label_template)

        with torch.enable_grad():

            for step in range(train_steps):

                adv_images01 = proj(image01s + delta)
                adv = image012pixel_values(adv_images01)
                
                img_feat = self.clip.get_image_features(pixel_values=adv).pooler_output # (N, 768)
                img_feat = F.normalize(img_feat, dim=-1)
                logit_scale = self.clip.logit_scale.exp()
                scores = logit_scale * (img_feat * target_feats).sum(dim=-1)  # (N,)

                loss = (- scores + gamma * delta.square().mean(dim=(1,2,3))).mean()


                print(f"Step {step} loss {loss.item()}")

                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    delta.copy_((image01s + delta).clamp(0, 1) - image01s)  # keep image in [0, 1]

        return proj(image01s + delta).detach().clamp(0, 1)
