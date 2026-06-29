from typing import Any
from pathlib import Path
import torch
import gc
import os

from .utils import (
    load_yaml_config,
    save_yaml_config,
    save_json,
    seed_everything,
    tee_console,
    collect_environment,
)

from .model import VLM_factory


def run(args: list | dict[str, Any] | str | Path) -> Any:
    '''
    configuration structure:


    -   output_dir: ...
        task_type: ...
        ...

    -   ...
    -   ...
        

    '''

    print(" >> Experiment Top-level")

    if isinstance(args, (str, Path)):
        print(" >> Load configurations from", args)
        args = load_yaml_config(args)

    if isinstance(args, list):
        # iterate through all tasks
        for item in args:
            run(item)

    else:
        
        output_dir = Path(args['output_dir'])

        # we require to start from a fresh directory
        os.makedirs(output_dir, exist_ok=False)

        save_yaml_config(args, output_dir / "config.yaml")

        # mirror everything printed to stdout/stderr into the output folder
        with tee_console(output_dir / "console.log"):

            # execute a single task
            if args["task_type"] == "VLM-ImageClass":
                res = task_VLM_ImageClass(args)

            else:
                raise ValueError("Invalid Config Type.")

            # sign a signature that the task is done
            with open(output_dir / "DONE", 'w') as p:
                p.write("THIS TASK FINISHED GRACEFULLY.")

        return res


def task_VLM_ImageClass(
    args: dict
):
    print(" >> Start VLM ImageClass Classification Task")

    from .model import VLM_factory
    from .bench import ImageClass_factory

    vlm_args = args['vlm']
    vlm = VLM_factory(vlm_args)

    imageclass_args = args['benchmark']
    imageclass = ImageClass_factory(imageclass_args)

    question = args['question']
    answer_priming = args['answer_priming']
    batch_size = args['batch_size']
    limit = args['limit']
    shuffle = args['shuffle']
    seed = args['seed']

    res = imageclass.eval_classify_lp(
        vlm,
        question,
        answer_priming,
        batch_size,
        limit,
        shuffle,
        seed
    )

    output_dir = Path(args['output_dir'])

    # split the result
    tensor_keys = ["indices", "preds", "correct_labels", "logprobs"]
    tensors = {k: res[k] for k in tensor_keys}
    summary = {k: v for k, v in res.items() if k not in tensor_keys}

    # readable summary
    save_json(summary, output_dir / "results.json")

    # per-example tensors for downstream analysis
    torch.save(tensors, output_dir / "tensors.pt")

    print(f" >> Saved results to {output_dir} (accuracy={res['accuracy']:.4f})")

    return res



    