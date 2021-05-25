import multiprocessing
import os
from multiprocessing.spawn import freeze_support

from dev_tools import shell_tools
from dev_tools.shell_tools import run_shell


def pip_compile(f: str):
    print(f"{f} started.")
    try:
        run_shell(f"pip-compile --upgrade --quiet {f}")
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
    in_files = [
        os.path.join("dev_tools/requirements", f)
        for f in os.listdir("dev_tools/requirements")
        if f.endswith('.in')
    ]
    in_files += [
        os.path.join("dev_tools/requirements/deps", f)
        for f in os.listdir("dev_tools/requirements/deps")
        if f.endswith('.in')
    ]

    freeze_support()
    pool = multiprocessing.Pool()

    try:
        pool.map(pip_compile, in_files)
    finally:
        pool.close()
        print("done")
