#!/usr/bin/env python

import shutil
import sys
import tempfile

from python_ci_utils import env_tools, all_checks


REPO_ORGANIZATIOn = 'quantumlib'
REPO_NAME = 'cirq'


def main():
    pull_request_number = None if len(sys.argv) == 1 else int(sys.argv[1])
    checks = all_checks.ALL_CHECKS
    access_token = None
    # pull_request_number = 222

    test_dir = tempfile.mkdtemp(prefix='test-{}-'.format(REPO_NAME))
    test_dir_2 = tempfile.mkdtemp(prefix='test-{}-py2-'.format(REPO_NAME))
    try:
        env = env_tools.prepare_temporary_test_environment(
            destination_directory=test_dir,
            repository=env_tools.GithubRepository(
                organization=REPO_ORGANIZATIOn,
                name=REPO_NAME,
                access_token=access_token),
            pull_request_number=pull_request_number)

        env2 = env_tools.derive_temporary_python2_environment(
            destination_directory=test_dir_2,
            python3_environment=env)

        results = [(check.context(), check.run_and_report(env, env2))
                   for check in checks]
    finally:
        shutil.rmtree(test_dir)
        shutil.rmtree(test_dir_2)

    print("ALL CHECK RESULTS")
    for result in results:
        print(result)

    for _, (_, _, error) in results:
        if error is not None:
            raise EnvironmentError('At least one check raised.') from error


if __name__ == '__main__':
    main()
