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

import subprocess
import sys
from typing import List, Tuple, Union


BOLD = 1
DIM = 2
RED = 31
GREEN = 32
YELLOW = 33


def highlight(text: str, color_code: int, bold: bool = False) -> str:
    """Wraps the given string with terminal color codes.

    Args:
        text: The content to highlight.
        color_code: The color to highlight with, e.g. 'shelltools.RED'.
        bold: Whether to bold the content in addition to coloring.

    Returns:
        The highlighted string.
    """
    return '{}\033[{}m{}\033[0m'.format('\033[1m' if bold else '', color_code, text)


def abbreviate_command_arguments_after_switches(cmd: Tuple[str, ...]) -> Tuple[str, ...]:
    result = [cmd[0]]
    for i in range(1, len(cmd)):
        if not cmd[i].startswith('-'):
            result.append('[...]')
            break
        result.append(cmd[i])
    return tuple(result)


def run(
    args: Union[str, List[str]],
    *,
    log_run_to_stderr: bool = True,
    abbreviate_non_option_arguments: bool = False,
    check: bool = True,
    text: bool = True,
    **subprocess_run_kwargs,
) -> subprocess.CompletedProcess:
    """Call subprocess.run with an option to log executed command to stderr.

    Args:
        args: The arguments for launching the process.  This may be a list
            or a string.  The string type may need to be used with
            ``shell=True`` to allow invocation as a shell command;
            otherwise the string is used as a command name with no arguments.
        log_run_to_stderr: Determines whether the fact that this command
            was executed is logged to sys.stderr or not.
        abbreviate_non_option_arguments: When logging to stderr, this cuts off
            the potentially-huge tail of the command listing off e.g. hundreds
            of file paths. No effect if log_run_to_stderr is not set.
        check: Raise the CalledProcessError exception if this flag is
            set and the process returns a non-zero exit code.  This sets
            the default check argument for the `subprocess.run` to True.
        text: Use text mode for the stdout and stderr streams from the
            process.  This changes the default text argument to the
            `subprocess.run` call to True.
        **subprocess_run_kwargs: Arguments passed to `subprocess.run`.
            See `subprocess.run` for a full detail of supported arguments.

    Returns:
        subprocess.CompletedProcess: The return value from `subprocess.run`.

    Raises:
        subprocess.CalledProcessError: The process returned a non-zero error
            code and the check argument was set.
    """
    # setup our default for subprocess.run flag arguments
    subprocess_run_kwargs.update(check=check, text=text)
    if log_run_to_stderr:
        cmd_desc: Tuple[str, ...] = (args,) if isinstance(args, str) else tuple(args)
        if abbreviate_non_option_arguments:
            cmd_desc = abbreviate_command_arguments_after_switches(cmd_desc)
        print('run:', cmd_desc, file=sys.stderr)
    return subprocess.run(args, **subprocess_run_kwargs)


def output_of(args: Union[str, List[str]], **kwargs) -> str:
    """Invokes a subprocess and returns its output as a string.

    Args:
        args: The arguments for launching the process.  This may be a list
            or a string, for example, ["echo", "dog"] or "pwd".  The string
            type may need to be used with ``shell=True`` to allow invocation
            as a shell command, otherwise the string is used as a command name
            with no arguments.
        **kwargs: Extra arguments for the shell_tools.run function, such as
            a cwd (current working directory) argument.

    Returns:
        str: The standard output of the command with the last newline removed.

    Raises:
         subprocess.CalledProcessError: The process returned a non-zero error
            code and the `check` flag was True (default).
    """
    result = run(args, log_run_to_stderr=False, stdout=subprocess.PIPE, **kwargs).stdout

    # Strip final newline.
    if result.endswith('\n'):
        result = result[:-1]

    return result
