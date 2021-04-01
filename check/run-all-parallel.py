import curses
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from subprocess import Popen, PIPE
from typing import Dict, List, Sequence, Tuple

import duet


def cd_repo_root():
    "Find the git top-level path and change to that directory."
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    repo_root = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'], encoding='utf-8', capture_output=True
    ).stdout.strip()
    os.chdir(repo_root)
    return repo_root


@dataclass
class Result:
    """Simple wrapper around the result of a check."""

    success: bool
    stdout: str
    stderr: str


async def run(command: List[str], varname: str, results: Dict[str, Result]):
    """Run a check"""
    assert varname not in results
    p = Popen(command, stdout=PIPE, stderr=PIPE, encoding='utf-8')

    while True:
        await duet.completed_future(None)
        ret = p.poll()
        if ret is not None:
            break

    stdout, stderr = p.communicate()
    assert varname not in results
    results[varname] = Result(
        success=p.returncode == 0,
        stdout=stdout,
        stderr=stderr,
    )


class FailFastException(RuntimeError):
    """Raised when a task has failed and we want to stop other tasks."""


async def check(check_names: List[str], results: Dict[str, Result], fail_fast: bool):
    """Check the status of running checks.

    Use ncurses WIN to nicely print a status of all the running checks.
    """
    name_len = max(len(s) for s in check_names)
    while True:
        alldone = True
        anyfailed = False
        for i, name in enumerate(check_names):

            if name in results:
                if results[name].success:
                    status = 'Done!'
                else:
                    status = '*Fail*'
                    anyfailed = True
            else:
                alldone = False
                status = '(running)'
            WIN.addstr(i + HEADER_SIZE, 0, f'{name:{name_len}s} {status:9s}')

        WIN.addstr(len(check_names) + HEADER_SIZE, 0, '')  # position cursor
        WIN.refresh()
        if alldone:
            return

        if fail_fast and anyfailed:
            await duet.failed_future(FailFastException)
        else:
            await duet.completed_future(None)


# curses global variables
HEADER_SIZE = 1
WIN = None


async def run_checks(
    checks: Sequence[Tuple[str, Sequence[str]]], check_names: Sequence[str], fail_fast: bool
) -> Dict[str, Result]:
    """Run all the checks.

    If `fail_fast` is set to True, we return after the first failure.

    This function requires that `WIN` has been set up.
    """
    results: Dict[str, Result] = {}
    start = time.time()
    try:
        async with duet.new_scope() as scope:
            for name, cmd in checks:
                scope.spawn(run, cmd, name, results)

            scope.spawn(check, check_names, results, fail_fast)
        end = time.time()
        WIN.addstr(f'Finished in {end - start}s.\n')
    except FailFastException:
        end = time.time()
        WIN.addstr(f'Stopped after {end - start}s due to failed check.\n')

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
        if result.success:
            continue

        print()
        print('-' * 10 + f' {name} ' + '-' * 10)
        if result.stderr is not None:
            print(result.stderr)
        if result.stdout is not None:
            print(result.stdout)


def print_summary(results, check_names):
    """For all check, print a one-line summary of their status."""
    name_len = max(len(s) for s in check_names)
    for name in check_names:
        if name not in results:
            print(f'{name:{name_len}s} - incomplete')
            continue

        result = results[name]
        if result.success:
            print(f'{name:{name_len}s} - Success!')
        else:
            print(f'{name:{name_len}s} - *Fail*')


async def main(fail_fast=True):
    """The main function."""
    global WIN
    repo_root = cd_repo_root()
    checks = [
        ('mypy', ['check/mypy']),
        ('nbformat', ['check/nbformat']),
        ('format-incremental', ['check/format-incremental']),
        ('misc', ['check/misc']),
        ('pylint-changed-files', ['check/pylint-changed-files']),
        ('pylint', ['check/pylint']),
        ('pytest-changed-files', ['check/pytest-changed-files']),
        (
            'pytest-changed-files-and-incremental-coverage',
            ['check/pytest-changed-files-and-incremental-coverage'],
        ),
        ('pytest', ['check/pytest']),
        ('pytest-and-incremental-coverage', ['check/pytest-and-incremental-coverage']),
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

        results = await run_checks(checks, check_names, fail_fast)
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
    duet.run(main, **_parse_args())
