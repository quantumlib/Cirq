# coverage: ignore
import asyncio
import curses
import os
import re
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum
from subprocess import PIPE
from typing import Dict, Sequence, Tuple


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


class MsgType(Enum):
    """Types of messages to be collated in `run`'s Queue."""

    DONE = 'done'
    EARLY_TERMINATE = 'early_terminate'
    STDOUT = 'stdout'
    STDERR = 'stderr'


@dataclass
class Result:
    """Simple wrapper around the result of a check."""

    status: Enum
    stdout: str
    stderr: str


async def run(command: Sequence[str], early_terminate: asyncio.Event):
    """Run a check.

    Args:
        command: The list of tokens to run with `subprocess.Popen`.
    """
    p = await asyncio.create_subprocess_exec(*command, stdout=PIPE, stderr=PIPE)

    stdouts = []
    stderrs = []
    queue = asyncio.Queue(1)

    async def drain(aiter, msgtype: MsgType):
        async for item in aiter:
            await queue.put((msgtype, item))

    async def notifier(awaitable, msg):
        await awaitable
        await queue.put((msg, None))

    asyncio.create_task(drain(p.stdout, MsgType.STDOUT))
    asyncio.create_task(drain(p.stderr, MsgType.STDERR))
    asyncio.create_task(notifier(p.wait(), MsgType.DONE))
    asyncio.create_task(notifier(early_terminate.wait(), MsgType.EARLY_TERMINATE))

    # Poll for completion.
    msgtype: MsgType
    while True:
        msgtype, item = await queue.get()
        if msgtype == MsgType.DONE:
            break
        if msgtype == MsgType.EARLY_TERMINATE:
            p.terminate()

            # Otherwise, the pipes will be closed after the event loop is closed.
            p._transport.close()

            yield Result(
                status=Status.CANCELED,
                stdout='',
                stderr='',
            )
            return

        std_out_or_err = item.decode()
        if msgtype == MsgType.STDOUT:
            stdouts.append(std_out_or_err)
        elif msgtype == MsgType.STDERR:
            stderrs.append(std_out_or_err)
        else:
            raise AssertionError()
        yield std_out_or_err

    yield Result(
        status=Status.SUCCESS if p.returncode == 0 else Status.FAIL,
        stdout=''.join(stdouts),
        stderr=''.join(stderrs),
    )


def check(check_names: Sequence[str], results: Dict[str, Result], outs: Dict[str, str]) -> bool:
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
                status = '(cancel)'
        else:
            status = '(running)'
        out = outs.get(name, '')
        # https://stackoverflow.com/a/33925425
        out = re.sub(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]', '', out)
        status_len = name_len + 1 + 9 + 1
        chars_left = curses.COLS - status_len
        out = out[:chars_left]
        WIN.addstr(i + HEADER_SIZE, 0, f'{name:{name_len}s} {status:9s} {out:{chars_left}s}')

    WIN.addstr(len(check_names) + HEADER_SIZE, 0, '')  # position cursor
    WIN.refresh()

    return anyfailed


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
    outs: Dict[str, str] = {}
    early_terminate = asyncio.Event()
    start = time.time()

    async def drain(aiter, name):
        async for item in aiter:
            if isinstance(item, Result):
                results[name] = item
            else:
                outs[name] = item

            anyfailed = check(check_names, results, outs)
            if fail_fast and anyfailed:
                early_terminate.set()

    fofo = [
        asyncio.create_task(drain(run(cmd, early_terminate), name), name=name)
        for name, cmd in checks
    ]
    await asyncio.gather(*fofo)

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
            print(f'{name:{name_len}s} - (cancel)')
        else:
            raise ValueError()


def main(fail_fast=True, long_tests=True):
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
    ]
    if long_tests:
        checks += [
            ('pylint', ['check/pylint']),
            ('pytest', ['check/pytest', '-v']),
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

        results = asyncio.run(run_checks(checks, check_names, fail_fast))
    finally:
        pass
        curses.nocbreak()
        WIN.keypad(False)
        curses.echo()
        curses.endwin()

    print_outputs(results, check_names)
    print('\n')
    print_summary(results, check_names)


def _parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument('--fail-fast', dest='fail_fast', action='store_true')
    parser.add_argument('--no-fail-fast', dest='fail_fast', action='store_false')
    parser.set_defaults(fail_fast=True)

    parser.add_argument('--long-tests', dest='long_tests', action='store_true')
    parser.add_argument('--no-long-tests', dest='long_tests', action='store_false')
    parser.set_defaults(long_tests=False)
    return vars(parser.parse_args())


if __name__ == '__main__':
    main(**_parse_args())
