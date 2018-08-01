#!/usr/bin/env python

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

import os
import sys

from dev_tools import all_checks, prepared_env, shell_tools


def main():
    if len(sys.argv) < 2:
        print(shell_tools.highlight(
            'Must specify a comparison branch '
            '(e.g. "origin/master" or "HEAD~1").',
            shell_tools.RED))
        sys.exit(1)
    comparison_branch = sys.argv[1]

    env = prepared_env.PreparedEnv(
        github_repo=None,
        actual_commit_id=None,  # local uncommitted files
        compare_commit_id=comparison_branch,
        destination_directory=os.getcwd(),
        virtual_env_path=None)
    check_results = [
        all_checks.pytest.run(env, False, set()),
        all_checks.incremental_coverage.run(env, False, set()),
    ]
    if any(not e.success for e in check_results):
        print(shell_tools.highlight(
            'Failed.',
            shell_tools.RED))
        sys.exit(1)


if __name__ == '__main__':
    main()
