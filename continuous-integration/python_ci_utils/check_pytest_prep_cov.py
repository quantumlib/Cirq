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

import subprocess

import os.path

from python_ci_utils import env_tools, check


class TestAndPrepareCoverageCheck(check.Check):
    """Checks if the code passes unit tests.

    As a side effect, produces coverage files used by the coverage checks.
    """

    def context(self):
        return 'pytest by maintainer'

    def perform_check_py2(self, env: env_tools.PreparedEnv):
        TestAndPrepareCoverageCheck._common_run_helper(env, coverage=False)

    def perform_check(self, env: env_tools.PreparedEnv):
        TestAndPrepareCoverageCheck._common_run_helper(env, coverage=True)

    @staticmethod
    def _common_run_helper(env: env_tools.PreparedEnv, coverage: bool):
        pytest_path = os.path.join(env.virtual_env_path, 'bin', 'pytest')
        target_path = os.path.join(env.destination_directory, 'cirq')
        try:
            result = env_tools.sub_run(pytest_path,
                                       target_path,
                                       '--cov' if coverage else '',
                                       capture_stdout=True,
                                       raise_error_if_process_fails=False)
            print(result)
        except subprocess.CalledProcessError:
            return False, 'Tests failed.'
        return True, 'Tests passed!'
