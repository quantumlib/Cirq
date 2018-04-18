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


class LintCheck(check.Check):
    """Checks if the code is up to snuff."""

    def context(self):
        return 'pylint by maintainer'

    def perform_check(self, env: env_tools.PreparedEnv):
        pylint_path = os.path.join(env.virtual_env_path, 'bin', 'pylint')
        target_path = os.path.join(env.destination_directory, 'cirq')
        try:
            result = env_tools.sub_run(
                pylint_path,
                target_path,
                '--reports=no',
                '--score=no',
                '--output-format=colorized',
                '--rcfile={}'.format(os.path.join(env.destination_directory,
                                                  'continuous-integration',
                                                  '.pylintrc')),
                capture_stdout=True,
                raise_error_if_process_fails=False)
            print(result)
        except subprocess.CalledProcessError:
            return False, 'Lint.'
        return True, 'No lint here!'
