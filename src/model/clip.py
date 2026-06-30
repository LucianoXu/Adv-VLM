import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast

from src.utils import resolve_device
from src.image import image012pixel_values


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
        differentiable: bool = False
    ) -> torch.Tensor:
        '''
        image01: (N, 3, 336, 336)
        return logits (N, #labels)
        '''
        ctx = torch.enable_grad if differentiable else torch.no_grad
        with ctx():
            pixel_values = image012pixel_values(image01s).to(self.device)
            
            img_feat = self.clip.get_image_features(pixel_values=pixel_values).pooler_output
            img_feat = F.normalize(img_feat, dim=-1)
            logit_scale = self.clip.logit_scale.exp()
            return logit_scale * img_feat @ text_feat.t()