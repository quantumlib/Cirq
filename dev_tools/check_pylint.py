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
from typing import cast

import re

import os.path
import sys

from dev_tools import env_tools, shell_tools, check


class LintCheck(check.Check):
    """Checks if the code is up to snuff."""

    def command_line_switch(self):
        return 'pylint'

    def context(self):
        return 'pylint'

    def perform_check(self, env: env_tools.PreparedEnv, verbose: bool):
        base_path = cast(str, env.destination_directory)
        rc_path = os.path.join(base_path,
                               'dev_tools',
                               'conf',
                               '.pylintrc')
        files = list(
            env_tools.get_unhidden_ungenerated_python_files(base_path))

        result = shell_tools.run_cmd(
            env.bin('pylint'),
            '--rcfile={}'.format(rc_path),
            *files,
            out=shell_tools.TeeCapture(sys.stdout),
            raise_on_fail=False,
            log_run_to_stderr=verbose,
            abbreviate_non_option_arguments=True)

        output = cast(str, result[0])
        passed = result[2] == 0
        if passed:
            return True, 'No lint here!'
        file_line_count = len(re.findall(r'\*' * 10, output))
        total_line_count = len([e for e in output.split('\n') if e.strip()])
        issue_count = total_line_count - file_line_count

        return False, '{} issues'.format(issue_count)
