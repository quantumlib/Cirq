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

from python_ci_utils import env_tools, incremental_coverage, check


class IncrementalCoverageCheck(check.Check):
    """Checks if touched lines are covered by tests.

    This check must run after the pytest check, because that check is what
    computes the coverage files used by this check.
    """
    def context(self):
        return 'incremental coverage by maintainer'

    def perform_check(self, env: env_tools.PreparedEnv):
        uncovered_count = incremental_coverage.check_for_uncovered_lines(env)
        if not uncovered_count:
            return True, 'All covered!'
        return False, '{} touched uncovered lines'.format(uncovered_count)
