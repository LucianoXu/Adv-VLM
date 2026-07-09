from typing import Any

from .interface import VLM


def VLM_factory(vlm_args: dict) -> VLM:

    print(" >> Creating VLM with arguments: ", vlm_args)

    vlm_name = vlm_args['name']

    if vlm_name == "LLaVA":

        from .llava import LLaVA
        device = vlm_args['device']
        return LLaVA(device=device)

    elif vlm_name == "VisualRWKV":

        from .visualrwkv import VisualRWKV
        device = vlm_args['device']
        model_path = vlm_args['model_path']
        arch = vlm_args.get('arch')
        return VisualRWKV(device=device, model_path=model_path, arch=arch)

    else:
        raise ValueError("Invalid VLM name:", vlm_name)