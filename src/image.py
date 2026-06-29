"""
Image preprocessing for openai/clip-vit-large-patch14-336
"""

from transformers import CLIPImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from PIL import Image
import torch

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
    img_preproc = get_img_preproc()

    res = img_preproc(
        images = imgs,
        do_resize = False,
        do_center_crop = False,
        do_rescale = True,
        do_normalize = False,
        return_tensors="pt"
    )['pixel_values']

    return res

def image012pixel_values(imgs: torch.Tensor) -> torch.Tensor:
    img_preproc = get_img_preproc()

    res = img_preproc(
        images = imgs,
        do_resize = False,
        do_center_crop = False,
        do_rescale = False,
        do_normalize = True,
        return_tensors="pt"
    )['pixel_values']

    return res

def image012resized(ts: torch.Tensor) -> list[Image.Image]:
    res = []
    for t in ts:
        t = t.detach().cpu().float()
        array = (t * 255).clamp(0, 255).round().to(torch.uint8)
        res.append(Image.fromarray(array.permute(1, 2, 0).numpy()))

    return res
