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

from typing import Set, List, Iterable

import os
import shutil
import sys
import tempfile

from dev_tools import env_tools, shell_tools, all_checks, check


REPO_ORGANIZATION = 'quantumlib'
REPO_NAME = 'cirq'


def report_pending(env, checks, out_still_pending: Set[check.Check]):
    for c in checks:
        env.report_status_to_github('pending', 'Preparing...', c.context())
        out_still_pending.add(c)


def topologically_sorted_checks_with_deps(checks: Iterable[check.Check]
                                          ) -> List[check.Check]:
    result = []
    seen = set()  # type: Set[check.Check]

    def handle(item: check.Check):
        if item in seen:
            return
        seen.add(item)

        # Dependencies first.
        for dep in item.dependencies:
            handle(dep)

        result.append(item)

    for e in checks:
        handle(e)

    return result


def parse_args():
    args = sys.argv
    verbose = '--verbose' in args
    only = [e.split('--only=')[1]
            for e in args
            if e.startswith('--only=')]
    checks = all_checks.ALL_CHECKS
    if only:
        checks = [e for e in checks if e.command_line_switch() in only]
        if len(checks) != len(only):
            print('Bad --only argument. Allowed values {!r}.'.format(
                      [e.command_line_switch() for e in all_checks.ALL_CHECKS]),
                  file=sys.stderr)
            sys.exit(1)
    checks = topologically_sorted_checks_with_deps(checks)

    positionals = [arg for arg in args if not arg.startswith('-')]
    pull_request_number = None if len(positionals) < 2 else int(positionals[1])
    access_token = None if len(positionals) < 3 else int(positionals[2])
    if access_token is None:
        access_token = os.getenv('CIRQ_GITHUB_ACCESS_TOKEN')
    return pull_request_number, access_token, verbose, checks


def main():
    pull_request_number, access_token, verbose, checks = parse_args()
    if pull_request_number is None:
        print(shell_tools.highlight(
            'No pull request number given. Using local files.',
            shell_tools.YELLOW))
        print()

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
            commit_ids_known_callback=lambda e:
                report_pending(e, checks, currently_pending),
            verbose=verbose)

        env2 = None

        check_results = []
        failures = set()
        for c in checks:
            # Prepare environment if needed.
            if c.needs_python2_env() and env2 is None:
                env2 = env_tools.derive_temporary_python2_environment(
                    destination_directory=test_dir_2,
                    python3_environment=env,
                    verbose=verbose)

            # Run the check.
            print()
            result = c.pick_env_and_run_and_report(env, env2, verbose, failures)

            # Record results.
            check_results.append(result)
            currently_pending.remove(c)
            if not result.success:
                failures.add(c)
            print()

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(test_dir_2, ignore_errors=True)
        for c in currently_pending:
            if env is not None:
                env.report_status_to_github('error',
                                            'Unexpected error.',
                                            c.context())

    print()
    print("ALL CHECK RESULTS")
    for result in check_results:
        print(result)

    for result in check_results:
        if result.unexpected_error is not None:
            raise EnvironmentError('At least one check raised.') from (
                result.unexpected_error)

    if any(not e.success for e in check_results):
        sys.exit(1)


if __name__ == '__main__':
    main()
