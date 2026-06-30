"""
Image preprocessing for openai/clip-vit-large-patch14-336
"""

from transformers import CLIPImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from PIL import Image
import numpy as np
import torch

# CLIP normalization constants as (1, 3, 1, 1) tensors for broadcasting over (N, C, H, W)
CLIP_MEAN = torch.tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
CLIP_STD = torch.tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

# concepts:
# raw image (the original image in Image.Image type)
# -> resized image (the resized image, in Image.Image type) (336*336 PIL, uint8 [0, 255])
# -> image01 (the tensor version of resized image) (336*336 float CHW, [0, 1])
# -> pixel_values (the normalized image01 that is feed into models directly)


IMAGE_SIZE = 336

# the global image preprocessor instance
img_preprocessor = None

def get_img_preproc() -> CLIPImageProcessor:
    """
    preprocessing pipeline:

    resize shortest edge -> 336 (bicubic)
    center crop          -> 336 x 336
    rescale              -> * 1/255
    normalize            -> (x - CLIP_mean) / CLIP_std
    """
    global img_preprocessor

    if img_preprocessor is None:
        img_preprocessor = CLIPImageProcessor(
            do_resize=True,
            size={"shortest_edge": IMAGE_SIZE},
            resample=PILImageResampling.BICUBIC,        # == 3
            do_center_crop=True,
            crop_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
            do_rescale=True,
            rescale_factor=1 / 255,
            do_normalize=True,
            image_mean=OPENAI_CLIP_MEAN,
            image_std=OPENAI_CLIP_STD,
            do_convert_rgb=True,
        )

    return img_preprocessor


def raw2resized(imgs: Image.Image | list[Image.Image]) -> list[Image.Image]: 
    '''
    preprocess the images by resizing, so that the output images only need to be normalized before fed into the model.
    '''
    img_preproc = get_img_preproc()

    res = img_preproc(
        images = imgs,
        do_resize = True,
        do_center_crop = True,
        do_rescale = False,
        do_normalize = False
    )['pixel_values']

    return [Image.fromarray(array.permute(1, 2, 0).numpy()) for array in res]

def resized2image01(imgs: Image.Image | list[Image.Image]) -> torch.Tensor:
    if isinstance(imgs, Image.Image):
        imgs = [imgs]

    tensors = []
    for im in imgs:
        arr = np.array(im.convert("RGB"))                   # (H, W, 3) uint8 (writable copy)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        tensors.append(t)

    return torch.stack(tensors, dim=0)

def image012pixel_values(imgs: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.to(device=imgs.device, dtype=imgs.dtype)
    std = CLIP_STD.to(device=imgs.device, dtype=imgs.dtype)
    return (imgs - mean) / std

def quantize_ste(x: torch.Tensor) -> torch.Tensor:
    # forward: snap to the uint8 grid and clamp to [0, 1] -- exactly the image gen() will receive
    # backward: identity (straight-through), so gradients still flow to delta

    # AI generated
    xq = (x.clamp(0, 1) * 255).round() / 255
    return x + (xq - x).detach()


def image012resized(ts: torch.Tensor) -> list[Image.Image]:
    res = []
    for t in ts:
        t = t.detach().cpu().float()
        array = (t * 255).clamp(0, 255).round().to(torch.uint8)
        res.append(Image.fromarray(array.permute(1, 2, 0).numpy()))

    return res
