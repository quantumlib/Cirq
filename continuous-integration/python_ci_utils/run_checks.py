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

from typing import Set

import os
import shutil
import sys
import tempfile

from python_ci_utils import env_tools, all_checks, check


REPO_ORGANIZATION = 'quantumlib'
REPO_NAME = 'cirq'


def report_pending(env, checks, out_still_pending: Set[check.Check]):
    for c in checks:
        env.report_status_to_github('pending', 'Preparing...', c.context())
        out_still_pending.add(c)


def main():
    pull_request_number = None if len(sys.argv) < 2 else int(sys.argv[1])
    checks = all_checks.ALL_CHECKS
    access_token = None if len(sys.argv) < 3 else int(sys.argv[2])
    if access_token is None:
        access_token = os.getenv('CIRQ_GITHUB_ACCESS_TOKEN')

    test_dir = tempfile.mkdtemp(prefix='test-{}-'.format(REPO_NAME))
    test_dir_2 = tempfile.mkdtemp(prefix='test-{}-py2-'.format(REPO_NAME))
    currently_pending = set()
    env = None
    try:
        env = env_tools.prepare_temporary_test_environment(
            destination_directory=test_dir,
            repository=env_tools.GithubRepository(
                organization=REPO_ORGANIZATION,
                name=REPO_NAME,
                access_token=access_token),
            pull_request_number=pull_request_number,
            commit_ids_known_callback=
                lambda e: report_pending(e, checks, currently_pending))

        env2 = None

        results = []
        for c in checks:
            if c.needs_python2_env() and env2 is None:
                env2 = env_tools.derive_temporary_python2_environment(
                    destination_directory=test_dir_2,
                    python3_environment=env)
            currently_pending.remove(c)
            result = c.context(), c.run_and_report(env, env2)
            results.append(result)

    finally:
        shutil.rmtree(test_dir)
        shutil.rmtree(test_dir_2)
        for c in currently_pending:
            if env is not None:
                env.report_status_to_github('error',
                                            'Unexpected error.',
                                            c.context())

    print("ALL CHECK RESULTS")
    for result in results:
        print(result)

    for _, (_, _, error) in results:
        if error is not None:
            raise EnvironmentError('At least one check raised.') from error


if __name__ == '__main__':
    main()
