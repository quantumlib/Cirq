import multiprocessing
import os
import sys
from multiprocessing.spawn import freeze_support
from typing import Tuple

from dev_tools import shell_tools
from dev_tools.shell_tools import run_shell


def pip_compile(inout: Tuple[str, str]):
    f, output = inout
    print(f"{f} started.")
    try:
        run_shell(f"pip-compile --upgrade --quiet {f} -o {output}")
        print(
            shell_tools.highlight(
                f"{f} done.",
                shell_tools.GREEN,
            )
        )
    except BaseException as e:
        # no need to print the exception, pip-compile will print to the stderr
        print(
            shell_tools.highlight(
                f"{f} FAILED.",
                shell_tools.RED,
            )
        )
        raise e


if __name__ == '__main__':
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python version: {pyver}")

    input_dir = "dev_tools/requirements/src"
    output_dir = f"dev_tools/requirements/dist/{pyver}"
    os.makedirs(output_dir, exist_ok=True)
    in_files = [
        (os.path.join(input_dir, f), os.path.join(output_dir, f))
        for f in os.listdir(input_dir)
        if f.endswith('.in')
    ]
    input_dir = f"{input_dir}/deps"
    output_dir = f"{output_dir}/deps"
    os.makedirs(output_dir, exist_ok=True)
    in_files += [
        (os.path.join(input_dir, f), os.path.join(output_dir, f))
        for f in os.listdir(input_dir)
        if f.endswith('.in')
    ]

    freeze_support()
    pool = multiprocessing.Pool()

    try:
        pool.map(pip_compile, in_files)
    finally:
        pool.close()
        print("done")
