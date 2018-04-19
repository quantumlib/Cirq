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
from typing import Tuple

import re

import os.path

from python_ci_utils import env_tools, check


class TestAndPrepareCoverageCheck(check.Check):
    """Checks if the code passes unit tests.

    As a side effect, produces coverage files used by the coverage checks.
    """

    def context(self):
        return 'pytest by maintainer'

    def needs_python2_env(self):
        return True

    def perform_check_py2(self, env: env_tools.PreparedEnv):
        return TestAndPrepareCoverageCheck._common_run_helper(env,
                                                              coverage=False)

    def perform_check(self, env: env_tools.PreparedEnv):
        return TestAndPrepareCoverageCheck._common_run_helper(env,
                                                              coverage=True)

    @staticmethod
    def _common_run_helper(env: env_tools.PreparedEnv,
                           coverage: bool) -> Tuple[bool, str]:
        pytest_path = os.path.join(env.virtual_env_path, 'bin', 'pytest')
        target_path = os.path.join(env.destination_directory, 'cirq')
        result = env_tools.sub_run(
            pytest_path,
            target_path,
            '--cov' if coverage else '',
            '--cov-report=annotate' if coverage else '',
            capture_stdout=True,
            raise_error_if_process_fails=False)

        output = result[0]
        passed = result[2] == 0
        if passed:
            return True, 'Tests passed!'

        last_line = [e for e in output.split('\n') if e.strip()][-1]
        fail_match = re.match('.+=== (\d+) failed', last_line)
        if fail_match is None:
            return False, 'Tests failed.'
        return False, '{} tests failed.'.format(fail_match.group(1))
