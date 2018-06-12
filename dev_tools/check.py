# Copyright 2018 Google LLC
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

from typing import Tuple, Optional, cast

import abc
import os.path

from dev_tools import env_tools


class CheckResult:
    """Output of a status check that passed, failed, or error'ed."""
    def __init__(self,
                 check: 'Check',
                 success: bool,
                 message: str,
                 unexpected_error: Optional[Exception]) -> None:
        self.check = check
        self.success = success
        self.message = message
        self.unexpected_error = unexpected_error

    def __str__(self):
        return str((self.check.context(),
                    self.success,
                    self.message,
                    self.unexpected_error))


class Check(metaclass=abc.ABCMeta):
    """A status check that can performed in a python environment."""

    def __init__(self, *dependencies):
        self.dependencies = dependencies

    @abc.abstractmethod
    def command_line_switch(self):
        """Used to identify this check from the command line."""
        pass

    @abc.abstractmethod
    def context(self) -> str:
        """The name of this status check, as shown on github."""
        pass

    @abc.abstractmethod
    def perform_check(self,
                      env: env_tools.PreparedEnv,
                      verbose: bool) -> Tuple[bool, str]:
        """Evaluates the status check and returns a pass/fail with message.

        Args:
            env: Describes a prepared python 3 environment in which to run.
            verbose: When set, more progress output is produced.

        Returns:
            A tuple containing a pass/fail boolean and then a details message.
        """
        pass

    def needs_python2_env(self):
        return False

    def run_and_report(self,
                       env: env_tools.PreparedEnv,
                       env_py2: Optional[env_tools.PreparedEnv],
                       verbose: bool,
                       ) -> CheckResult:
        """Evaluates this check in python 3 and 2.7, and reports to github.

        If the prepared environments are not linked to a github repository,
        with a known access token, reporting to github is skipped.

        Args:
            env: A prepared python 3 environment.
            env_py2: A prepared python 2.7 environment.
            verbose: When set, more progress output is produced.

        Returns:
            A (success, message, error) tuple. The success element is True if
            the check passed, and False if the check failed or threw an
            unexpected exception. The message element is a human-readable
            blurb of what went right/wrong (used when reporting to github). The
            error element contains any unexpected exception thrown by the
            check. The caller is responsible for propagating this error
            appropriately.
        """
        env.report_status_to_github('pending', 'Running...', self.context())
        try:
            chosen_env = cast(env_tools.PreparedEnv,
                              env_py2 if self.needs_python2_env() else env)
            os.chdir(cast(str, chosen_env.destination_directory))
            success, message = self.perform_check(chosen_env, verbose=verbose)

        except Exception as ex:
            env.report_status_to_github('error',
                                        'Unexpected error.',
                                        self.context())
            return CheckResult(self, False, 'Unexpected error.', ex)

        env.report_status_to_github(
            'success' if success else 'failure',
            message,
            self.context())
        return CheckResult(self, success, message, None)
