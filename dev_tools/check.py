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

from typing import Tuple, Optional, cast, Set

import abc
import os.path

from dev_tools import env_tools, shell_tools


class CheckResult:
    """Output of a status check that passed, failed, or error'ed."""

    def __init__(
        self, check: 'Check', success: bool, message: str, unexpected_error: Optional[Exception]
    ) -> None:
        self.check = check
        self.success = success
        self.message = message
        self.unexpected_error = unexpected_error

    def __str__(self):
        outcome = 'ERROR' if self.unexpected_error else 'pass' if self.success else 'FAIL'
        msg = self.unexpected_error if self.unexpected_error else self.message
        result = f'{outcome}: {self.check.context()} ({msg})'
        return shell_tools.highlight(result, shell_tools.GREEN if self.success else shell_tools.RED)


class Check(metaclass=abc.ABCMeta):
    """A status check that can performed in a python environment."""

    def __init__(self, *dependencies):
        self.dependencies = dependencies

    @abc.abstractmethod
    def command_line_switch(self) -> str:
        """Used to identify this check from the command line."""

    @abc.abstractmethod
    def context(self) -> str:
        """The name of this status check, as shown on github."""

    @abc.abstractmethod
    def perform_check(self, env: env_tools.PreparedEnv, verbose: bool) -> Tuple[bool, str]:
        """Evaluates the status check and returns a pass/fail with message.

        Args:
            env: Describes a prepared python 3 environment in which to run.
            verbose: When set, more progress output is produced.

        Returns:
            A tuple containing a pass/fail boolean and then a details message.
        """

    def needs_python2_env(self) -> bool:
        return False

    def run(
        self, env: env_tools.PreparedEnv, verbose: bool, previous_failures: Set['Check']
    ) -> CheckResult:
        """Evaluates this check.

        Args:
            env: The prepared python environment to run the check in.
            verbose: When set, more progress output is produced.
            previous_failures: Checks that have already run and failed.

        Returns:
            A CheckResult instance.
        """

        # Skip if a dependency failed.
        if previous_failures.intersection(self.dependencies):
            print(
                shell_tools.highlight('Skipped ' + self.command_line_switch(), shell_tools.YELLOW)
            )
            return CheckResult(self, False, 'Skipped due to dependency failing.', None)

        print(shell_tools.highlight('Running ' + self.command_line_switch(), shell_tools.GREEN))
        try:
            success, message = self.perform_check(env, verbose=verbose)
            result = CheckResult(self, success, message, None)
        except Exception as ex:
            result = CheckResult(self, False, 'Unexpected error.', ex)

        print(
            shell_tools.highlight(
                'Finished ' + self.command_line_switch(),
                shell_tools.GREEN if result.success else shell_tools.RED,
            )
        )
        if verbose:
            print(result)

        return result

    def pick_env_and_run_and_report(
        self, env: env_tools.PreparedEnv, verbose: bool, previous_failures: Set['Check']
    ) -> CheckResult:
        """Evaluates this check in python 3 or 2.7, and reports to github.

        If the prepared environments are not linked to a github repository,
        with a known access token, reporting to github is skipped.

        Args:
            env: A prepared python 3 environment.
            verbose: When set, more progress output is produced.
            previous_failures: Checks that have already run and failed.

        Returns:
            A CheckResult instance.
        """
        env.report_status_to_github('pending', 'Running...', self.context())
        os.chdir(cast(str, env.destination_directory))

        result = self.run(env, verbose, previous_failures)

        if result.unexpected_error is not None:
            env.report_status_to_github('error', 'Unexpected error.', self.context())
        else:
            env.report_status_to_github(
                'success' if result.success else 'failure', result.message, self.context()
            )

        return result
