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

from typing import Tuple, Optional

import abc
import os.path

from python_ci_utils import env_tools


class Check(metaclass=abc.ABCMeta):
    """A status check that can performed in a python environment."""

    @abc.abstractmethod
    def context(self) -> str:
        """The name of this status check, as shown on github."""
        pass

    @abc.abstractmethod
    def perform_check(self, env: env_tools.PreparedEnv) -> Tuple[bool, str]:
        """Evaluates the status check and returns a pass/fail with message.

        Args:
            env: Describes a prepared python 3 environment in which to run.

        Returns:
            A tuple containing a pass/fail boolean and then a details message.
        """
        pass

    def perform_check_py2(self, env: env_tools.PreparedEnv
                          ) -> Optional[Tuple[bool, str]]:
        """Evaluates the status check in python 2.7, if appropriate.

        Args:
            env: Describes a prepared python 2.7 environment in which to run.

        Returns:
            A tuple containing a pass/fail boolean and then a details message,
            or else None if this check does not need to pass on code that has
            been automatically translated into python 2.
        """
        return None

    @staticmethod
    def _merge_result(result1: Tuple[bool, str],
                      result2: Optional[Tuple[bool, str]]
                      ) -> Tuple[bool, str]:
        if result2 is None or not result1[0] or result2[0]:
            return result1
        return result2

    def run_and_report(self,
                       env: env_tools.PreparedEnv,
                       env_py2: env_tools.PreparedEnv
                       ) -> Tuple[bool, str, Optional[Exception]]:
        """Evaluates this check in python 3 and 2.7, and reports to github.

        If the prepared environments are not linked to a github repository,
        with a known access token, reporting to github is skipped.

        Args:
            env: A prepared python 3 environment.
            env_py2: A prepared python 2.7 environment.

        Returns:
            A (success, message, error) tuple. The success element is True if
            the check passed, and False if the check failed or threw an
            unexpected exception. The message element is a human-readable
            blurb of what went right/wrong (used when reporting to github). The
            error element contains any unexpected exception thrown by the
            check. The caller is responsible for propagating this error
            appropriately.
        """
        env.report_status('pending', 'Running...', self.context())
        try:
            os.chdir(env.destination_directory)
            result1 = self.perform_check(env)
            os.chdir(env_py2.destination_directory)
            result2 = self.perform_check_py2(env_py2)
            success, message = Check._merge_result(result1, result2)
        except Exception as ex:
            env.report_status('error', 'Unexpected error.', self.context())
            return False, 'Unexpected error.', ex

        env.report_status(
            'success' if success else 'failure',
            message,
            self.context())
        return success, message, None
