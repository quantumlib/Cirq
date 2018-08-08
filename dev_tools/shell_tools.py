# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import subprocess
import sys
from typing import (
    Optional, Tuple, Union, IO, Any, cast, TYPE_CHECKING, NamedTuple,
)

import collections

CommandOutput = NamedTuple(
    "CommandOutput",
    [
        ('out', Optional[str]),
        ('err', Optional[str]),
        ('exit_code', int),
    ]
)

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import List


BOLD = 1
DIM = 2
RED = 31
GREEN = 32
YELLOW = 33


def highlight(text: str, color_code: int, bold: bool=False) -> str:
    """Wraps the given string with terminal color codes.

    Args:
        text: The content to highlight.
        color_code: The color to highlight with, e.g. 'shelltools.RED'.
        bold: Whether to bold the content in addition to coloring.

    Returns:
        The highlighted string.
    """
    return '{}\033[{}m{}\033[0m'.format(
        '\033[1m' if bold else '',
        color_code,
        text,)


class TeeCapture:
    """Marker class indicating desire to capture output written to a pipe.

    If out_pipe is None, the caller just wants to capture output without
    writing it to anything in particular.
    """
    def __init__(self, out_pipe: Optional[IO[str]] = None) -> None:
        self.out_pipe = out_pipe


async def _async_forward(async_chunks: collections.AsyncIterable,
                         out: Optional[Union[TeeCapture, IO[str]]]
                         ) -> Optional[str]:
    """Prints/captures output from the given asynchronous iterable.

    Args:
        async_chunks: An asynchronous source of bytes or str.
        out: Where to put the chunks.

    Returns:
        The complete captured output, or else None if the out argument wasn't a
        TeeCapture instance.
    """
    capture = isinstance(out, TeeCapture)
    out_pipe = out.out_pipe if isinstance(out, TeeCapture) else out

    chunks = [] if capture else None  # type: Optional[List[str]]
    async for chunk in async_chunks:
        if not isinstance(chunk, str):
            chunk = chunk.decode()
        if out_pipe:
            print(chunk, file=out_pipe, end='')
        if chunks is not None:
            chunks.append(chunk)

    return ''.join(chunks) if chunks is not None else None


async def _async_wait_for_process(
        future_process: Any,
        out: Optional[Union[TeeCapture, IO[str]]] = sys.stdout,
        err: Optional[Union[TeeCapture, IO[str]]] = sys.stderr
) -> CommandOutput:
    """Awaits the creation and completion of an asynchronous process.

    Args:
        future_process: The eventually created process.
        out: Where to write stuff emitted by the process' stdout.
        err: Where to write stuff emitted by the process' stderr.

    Returns:
        A (captured output, captured error output, return code) triplet.
    """
    process = await future_process
    future_output = _async_forward(process.stdout, out)
    future_err_output = _async_forward(process.stderr, err)
    output, err_output = await asyncio.gather(future_output, future_err_output)
    await process.wait()

    return CommandOutput(output, err_output, process.returncode)


def abbreviate_command_arguments_after_switches(
        cmd: Tuple[str, ...]) -> Tuple[str, ...]:
    result = [cmd[0]]
    for i in range(1, len(cmd)):
        if not cmd[i].startswith('-'):
            result.append('[...]')
            break
        result.append(cmd[i])
    return tuple(result)


def run_cmd(*cmd: Optional[str],
            out: Optional[Union[TeeCapture, IO[str]]] = sys.stdout,
            err: Optional[Union[TeeCapture, IO[str]]] = sys.stderr,
            raise_on_fail: bool = True,
            log_run_to_stderr: bool = True,
            abbreviate_non_option_arguments: bool = False,
            **kwargs
            ) -> CommandOutput:
    """Invokes a subprocess and waits for it to finish.

    Args:
        cmd: Components of the command to execute, e.g. ["echo", "dog"].
        out: Where to write the process' stdout. Defaults to sys.stdout. Can be
            anything accepted by print's 'file' parameter, or None if the
            output should be dropped, or a TeeCapture instance. If a TeeCapture
            instance is given, the first element of the returned tuple will be
            the captured output.
        err: Where to write the process' stderr. Defaults to sys.stderr. Can be
            anything accepted by print's 'file' parameter, or None if the
            output should be dropped, or a TeeCapture instance. If a TeeCapture
            instance is given, the second element of the returned tuple will be
            the captured error output.
        raise_on_fail: If the process returns a non-zero error code
            and this flag is set, a CalledProcessError will be raised.
            Otherwise the return code is the third element of the returned
            tuple.
        log_run_to_stderr: Determines whether the fact that this shell command
            was executed is logged to sys.stderr or not.
        abbreviate_non_option_arguments: When logging to stderr, this cuts off
            the potentially-huge tail of the command listing off e.g. hundreds
            of file paths. No effect if log_run_to_stderr is not set.

        **kwargs: Extra arguments for asyncio.create_subprocess_shell, such as
            a cwd (current working directory) argument.

    Returns:
        A (captured output, captured error output, return code) triplet. The
        captured outputs will be None if the out or err parameters were not set
        to an instance of TeeCapture.

    Raises:
         subprocess.CalledProcessError: The process returned a non-zero error
            code and raise_on_fail was set.
    """
    kept_cmd = tuple(cast(str, e) for e in cmd if e is not None)
    if log_run_to_stderr:
        cmd_desc = kept_cmd
        if abbreviate_non_option_arguments:
            cmd_desc = abbreviate_command_arguments_after_switches(cmd_desc)
        print('run:', cmd_desc, file=sys.stderr)
    result = asyncio.get_event_loop().run_until_complete(
        _async_wait_for_process(
            asyncio.create_subprocess_exec(
                *kept_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **kwargs),
            out,
            err))
    if raise_on_fail and result[2]:
        raise subprocess.CalledProcessError(result[2], kept_cmd)
    return result


def run_shell(cmd: str,
              out: Optional[Union[TeeCapture, IO[str]]] = sys.stdout,
              err: Optional[Union[TeeCapture, IO[str]]] = sys.stderr,
              raise_on_fail: bool = True,
              log_run_to_stderr: bool = True,
              **kwargs
              ) -> CommandOutput:
    """Invokes a shell command and waits for it to finish.

    Args:
        cmd: The command line string to execute, e.g. "echo dog | cat > file".
        out: Where to write the process' stdout. Defaults to sys.stdout. Can be
            anything accepted by print's 'file' parameter, or None if the
            output should be dropped, or a TeeCapture instance. If a TeeCapture
            instance is given, the first element of the returned tuple will be
            the captured output.
        err: Where to write the process' stderr. Defaults to sys.stderr. Can be
            anything accepted by print's 'file' parameter, or None if the
            output should be dropped, or a TeeCapture instance. If a TeeCapture
            instance is given, the second element of the returned tuple will be
            the captured error output.
        raise_on_fail: If the process returns a non-zero error code
            and this flag is set, a CalledProcessError will be raised.
            Otherwise the return code is the third element of the returned
            tuple.
        log_run_to_stderr: Determines whether the fact that this shell command
            was executed is logged to sys.stderr or not.
        **kwargs: Extra arguments for asyncio.create_subprocess_shell, such as
            a cwd (current working directory) argument.

    Returns:
        A (captured output, captured error output, return code) triplet. The
        captured outputs will be None if the out or err parameters were not set
        to an instance of TeeCapture.

    Raises:
         subprocess.CalledProcessError: The process returned a non-zero error
            code and raise_on_fail was set.
    """
    if log_run_to_stderr:
        print('shell:', cmd, file=sys.stderr)
    result = asyncio.get_event_loop().run_until_complete(
        _async_wait_for_process(
            asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **kwargs),
            out,
            err))
    if raise_on_fail and result[2]:
        raise subprocess.CalledProcessError(result[2], cmd)
    return result


def output_of(*cmd: Optional[str], **kwargs) -> str:
    """Invokes a subprocess and returns its output as a string.

    Args:
        cmd: Components of the command to execute, e.g. ["echo", "dog"].
        **kwargs: Extra arguments for asyncio.create_subprocess_shell, such as
            a cwd (current working directory) argument.

    Returns:
        A (captured output, captured error output, return code) triplet. The
        captured outputs will be None if the out or err parameters were not set
        to an instance of TeeCapture.

    Raises:
         subprocess.CalledProcessError: The process returned a non-zero error
            code and raise_on_fail was set.
    """
    result = cast(str, run_cmd(*cmd,
                               log_run_to_stderr=False,
                               out=TeeCapture(),
                               **kwargs).out)

    # Strip final newline.
    if result.endswith('\n'):
        result = result[:-1]

    return result
