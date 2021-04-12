import concurrent
import curses
import os
import subprocess
import sys
import time
from concurrent.futures._base import Future
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from subprocess import Popen, PIPE
from typing import Dict, List, Sequence, Tuple

_EARLY_TERMINATE = False


def cd_repo_root():
    "Find the git top-level path and change to that directory."
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    repo_root = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], encoding='utf-8', capture_output=True
    ).stdout.strip()
    os.chdir(repo_root)
    return repo_root


class Status(Enum):
    """Possible results of a test."""

    SUCCESS = 'success'
    FAIL = 'fail'
    CANCELED = 'cancel'


@dataclass
class Result:
    """Simple wrapper around the result of a check."""

    status: Enum
    stdout: str
    stderr: str


def run(command: List[str]):
    """Run a check.

    Args:
        command: The list of tokens to run with `subprocess.Popen`.
    """
    p = Popen(command, stdout=PIPE, stderr=PIPE, encoding='utf-8')

    # Poll for completion.
    while True:
        ret = p.poll()
        if ret is not None:
            break

        # ThreadPoolExecutor does not provide a way to kill threads. Any tasks that have
        # already been submitted will run to completion. Here, we use a global variable
        # to flag that all outstanding checks should be forcibly terminated.
        if _EARLY_TERMINATE:
            p.terminate()

            # Return here so we can set the correct status and so it does not hang
            # on `p.communicate()` later.
            return Result(
                status=Status.CANCELED,
                stdout='',
                stderr='',
            )

    stdout, stderr = p.communicate()
    return Result(
        status=Status.SUCCESS if p.returncode == 0 else Status.FAIL,
        stdout=stdout,
        stderr=stderr,
    )


def check(check_names: Sequence[str], results: Dict[str, Result]) -> bool:
    """Check the status of running checks.

    Use ncurses WIN to nicely print a status of all the running checks.
    Returns whether any tasks failed.
    """
    name_len = max(len(s) for s in check_names)
    anyfailed = False
    for i, name in enumerate(check_names):

        if name in results:
            if results[name].status == Status.SUCCESS:
                status = 'Done!'
            elif results[name].status == Status.FAIL:
                status = '*Fail*'
                anyfailed = True
            else:
                status = '(cancelled)'
        else:
            status = '(running)'
        WIN.addstr(i + HEADER_SIZE, 0, f'{name:{name_len}s} {status:9s}')

    WIN.addstr(len(check_names) + HEADER_SIZE, 0, '')  # position cursor
    WIN.refresh()

    return anyfailed


# curses global variables
HEADER_SIZE = 1
WIN = None


def run_checks(
    checks: Sequence[Tuple[str, Sequence[str]]], check_names: Sequence[str], fail_fast: bool
) -> Dict[str, Result]:
    """Run all the checks.

    If `fail_fast` is set to True, we return after the first failure.

    This function requires that `WIN` has been set up.
    """
    global _EARLY_TERMINATE
    results: Dict[str, Result] = {}
    futures: Dict[Future, str] = {}
    start = time.time()
    with ThreadPoolExecutor() as executor:
        # Submit all the checks to the executor.
        for name, cmd in checks:
            fut = executor.submit(run, cmd)
            futures[fut] = name

        # Process the results as the become available.
        for done in concurrent.futures.as_completed(futures):
            result = done.result()
            name = futures[done]
            results[name] = result
            anyfailed = check(check_names, results)
            if fail_fast and anyfailed:
                _EARLY_TERMINATE = True
                executor.shutdown(wait=False)

        end = time.time()
        WIN.addstr(f'Finished in {end - start}s.\n')

    press_a_key = True
    if press_a_key:
        WIN.addstr('Press a key')
        WIN.getkey()
    return results


def print_outputs(results: Dict[str, Result], check_names: Sequence[str]):
    """For all failed checks, print stdout and stderr."""
    for name in check_names:
        if name not in results:
            continue
        result = results[name]
        if result.status != Status.FAIL:
            continue

        print()
        print('-' * 10 + f' {name} ' + '-' * 10)
        if result.stderr is not None:
            print(result.stderr)
        if result.stdout is not None:
            print(result.stdout)


def print_summary(results: Dict[str, Result], check_names: Sequence[str]):
    """For all checks, print a one-line summary of their status."""
    name_len = max(len(s) for s in check_names)
    for name in check_names:
        if name not in results:
            print(f'{name:{name_len}s} - incomplete')
            continue

        result = results[name]
        if result.status == Status.SUCCESS:
            print(f'{name:{name_len}s} - Success!')
        elif result.status == Status.FAIL:
            print(f'{name:{name_len}s} - *Fail*')
        elif result.status == Status.CANCELED:
            print(f'{name:{name_len}s} - (cancelled)')
        else:
            raise ValueError()


def main(fail_fast=True):
    """The main function."""
    global WIN
    repo_root = cd_repo_root()
    checks = [
        ('misc', ['check/misc']),
        ('mypy', ['check/mypy']),
        ('nbformat', ['check/nbformat']),
        ('format-incremental', ['check/format-incremental']),
        ('pylint-changed-files', ['check/pylint-changed-files']),
        ('pytest-changed-files', ['check/pytest-changed-files']),
        ('incremental-coverage', ['check/pytest-changed-files-and-incremental-coverage']),
        ('pylint', ['check/pylint']),
        ('pytest', ['check/pytest']),
    ]
    check_names = [name for name, _ in checks]

    try:
        WIN = curses.initscr()
        curses.setupterm()
        curses.noecho()
        curses.cbreak()
        WIN.keypad(True)

        WIN.addstr(0, 0, f'Repo root: {repo_root}')
        WIN.refresh()

        results = run_checks(checks, check_names, fail_fast)
    finally:
        curses.nocbreak()
        WIN.keypad(False)
        curses.echo()
        curses.endwin()

    print_outputs(results, check_names)
    print('\n')
    print_summary(results, check_names)


def _parse_args():
    """Parse command line arguments."""
    if len(sys.argv) == 1:
        # Default
        return {'fail_fast': True}
    if len(sys.argv) > 2:
        raise ValueError("Unknown command line arguments")

    if sys.argv[1] == '--fail-fast':
        return {'fail_fast': True}
    if sys.argv[1] == '--no-fail-fast':
        return {'fail_fast': False}

    raise ValueError("Unknown command line argument")


if __name__ == '__main__':
    main(**_parse_args())
