from typing import Any

from .interface import ImageClass

from .imagenette import Imagenette


def ImageClass_factory(bench_args: dict) -> ImageClass:

    print(" >> Creating benchmark with arguments: ", bench_args)

    bench_name = bench_args['name']

    if bench_name == "imagenette":
        return Imagenette(local_path=bench_args.get("local_path"))

    else:
        raise ValueError("Invalid benchmark name:", bench_name)