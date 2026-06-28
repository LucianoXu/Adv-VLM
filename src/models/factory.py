from typing import Any
from pathlib import Path
import torch

from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.utils import GenerationMixin

from ..utils import load_yaml_config

def model_factory(model_args: dict[str, Any] | str | Path) -> tuple[PreTrainedTokenizerBase, GenerationMixin]:

    print(" >> Model Factory for", model_args)

    if isinstance(model_args, (str, Path)):
        model_args = load_yaml_config(model_args)

    if model_args["model_name"] == "LLaVA":
        pass
    
    else:
        raise ValueError("Invalid Model Name: ", model_args["model_name"])
    