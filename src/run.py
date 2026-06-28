from typing import Any
from pathlib import Path
import torch
import gc

from .utils import (
    load_yaml_config,
    save_yaml_config,
    save_json,
    seed_everything,
    tee_console,
    collect_environment,
)

from .models.factory import model_factory


def run(args: dict[str, Any] | str | Path) -> Any:

    print(" >> Experiment Top-level")

    if isinstance(args, (str, Path)):
        print(" >> Load configurations from", args)
        args = load_yaml_config(args)


    if args["config_type"] == "model":
        return model_factory(args)
    
    else:
        raise ValueError("Invalid Config Type.")