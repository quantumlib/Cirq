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

from dev_tools import all_checks, prepared_env


def main():
    verbose = True
    checks = [
        all_checks.pytest,
        all_checks.incremental_coverage,
    ]

    env = prepared_env.PreparedEnv(
        github_repo=None,
        actual_commit_id=None,  # local uncommitted files
        compare_commit_id='master',
        destination_directory=os.getcwd(),
        virtual_env_path=None)
    check_results = []
    failures = set()
    for c in checks:
        print()
        result = c.run(env, verbose, failures)

        # Record results.
        check_results.append(result)
        if not result.success:
            failures.add(c)
        print()

    print()
    print("ALL CHECK RESULTS")
    for result in check_results:
        print(result)

    if any(not e.success for e in check_results):
        sys.exit(1)


if __name__ == '__main__':
    main()
