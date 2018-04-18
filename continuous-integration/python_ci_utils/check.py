#!/usr/bin/env python

import shutil
import tempfile

from python_ci_utils import env_tools, all_checks


REPOSITORY_ORGANIZATION = 'quantumlib'
REPOSITORY_NAME = 'cirq'

def main():
    checks = all_checks.ALL_CHECKS
    test_dir = tempfile.mkdtemp(prefix='test-{}'.format(REPOSITORY_NAME))
    try:
        env = env_tools.prepare_temporary_test_environment(
            destination_directory=test_dir,
            repository=env_tools.GithubRepository(
                organization=REPOSITORY_ORGANIZATION,
                name=REPOSITORY_NAME,
                access_token=None),
            pull_request_number=222)

        results = [(check.context(), check.run_and_report(env))
                   for check in checks]
    finally:
        shutil.rmtree(test_dir)

    print("ALL CHECK RESULTS")
    for result in results:
        print(result)

    for _, (_, _, error) in results:
        if error is not None:
            raise EnvironmentError('At least one check raised.') from error


if __name__ == '__main__':
    main()
