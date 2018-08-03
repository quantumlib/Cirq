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

import os
import re
import sys

from dev_tools import env_tools, shell_tools, check


class TestAndPrepareCoverageCheck(check.Check):
    """Checks if the code passes unit tests.

    As a side effect, produces coverage files used by the coverage checks.
    """

    def command_line_switch(self):
        return 'pytest'

    def context(self):
        return 'pytest'

    def perform_check(self, env: env_tools.PreparedEnv, verbose: bool):
        do_coverage = True
        base_path = cast(str, env.destination_directory)
        rc_path = os.path.join(base_path,
                               'dev_tools',
                               'conf',
                               '.coveragerc')
        target_path = base_path
        result = shell_tools.run_cmd(
            env.bin('pytest'),
            target_path,
            None if verbose else '--quiet',
            *([
                  '--cov',
                  '--cov-report=annotate',
                  '--cov-config={}'.format(rc_path)
              ] if do_coverage else []),
            out=shell_tools.TeeCapture(sys.stdout),
            raise_on_fail=False,
            log_run_to_stderr=verbose)

        output = cast(str, result[0])
        passed = result[2] == 0
        if passed:
            return True, 'Tests passed!'

        last_line = [e for e in output.split('\n') if e.strip()][-1]
        fail_match = re.match('.+=== (\d+) failed', last_line)
        if fail_match is None:
            return False, 'Tests failed.'
        return False, '{} tests failed.'.format(fail_match.group(1))
