from typing import Any

from .interface import VLM


def VLM_factory(vlm_args: dict) -> VLM:

    print(" >> Creating VLM with arguments: ", vlm_args)

    vlm_name = vlm_args['name']

    if vlm_name == "LLaVA":

        from .llava import LLaVA
        device = vlm_args['device']
        return LLaVA(device=device)
    
    else:
        raise ValueError("Invalid VLM name:", vlm_name)