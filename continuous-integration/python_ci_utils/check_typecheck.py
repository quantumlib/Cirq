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

import os.path

from python_ci_utils import env_tools, check


class TypeCheck(check.Check):
    """Checks that the types specified in the code make sense."""

    def context(self):
        return 'typecheck by maintainer'

    def perform_check(self, env: env_tools.PreparedEnv):
        mypy_path = os.path.join(env.virtual_env_path, 'bin', 'mypy')
        files = list(env_tools.get_unhidden_ungenerated_python_files(
            env.destination_directory))
        result = env_tools.sub_run(
            mypy_path,
            '--config-file={}'.format(os.path.join(
                env.destination_directory,
                'continuous-integration',
                'mypy.ini')),
            *files,
            capture_stdout=True,
            raise_error_if_process_fails=False)

        output = result[0]
        passed = result[2] == 0
        if passed:
            return True, 'Types look good!'
        issue_count = len([e for e in output.split('\n') if e.strip()])

        return False, '{} issues'.format(issue_count)
