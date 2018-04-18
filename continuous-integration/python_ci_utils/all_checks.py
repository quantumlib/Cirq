import subprocess
from typing import Tuple, Optional

import abc
import os.path

from python_ci_utils import env_tools, incremental_coverage


class Check(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def context(self) -> str:
        pass

    def run_py2_helper(self, env: env_tools.PreparedEnv
                       ) -> Optional[Tuple[bool, str]]:
        return None

    @abc.abstractmethod
    def run_helper(self, env: env_tools.PreparedEnv) -> Tuple[bool, str]:
        pass

    @staticmethod
    def _merge_result(result1: Tuple[bool, str],
                      result2: Optional[Tuple[bool, str]]
                      ) -> Tuple[bool, str]:
        if result2 is None or not result1[0] or result2[0]:
            return result1
        return result2

    def run_and_report(self,
                       env: env_tools.PreparedEnv,
                       env_py2: env_tools.PreparedEnv
                       ) -> Tuple[bool, str, Optional[Exception]]:
        env.report_status('pending', 'Running...', self.context())
        try:
            os.chdir(env.destination_directory)
            result1 = self.run_helper(env)
            os.chdir(env_py2.destination_directory)
            result2 = self.run_py2_helper(env_py2)
            success, message = Check._merge_result(result1, result2)
        except Exception as ex:
            env.report_status('error', 'Unexpected error.', self.context())
            return False, 'Unexpected error.', ex

        env.report_status(
            'success' if success else 'failure',
            message,
            self.context())
        return success, message, None


class TestAndPrepareCoverageCheck(Check):
    def context(self):
        return 'pytest by maintainer'

    def run_py2_helper(self, env: env_tools.PreparedEnv):
        TestAndPrepareCoverageCheck._common_run_helper(env, coverage=False)

    def run_helper(self, env: env_tools.PreparedEnv):
        TestAndPrepareCoverageCheck._common_run_helper(env, coverage=True)

    @staticmethod
    def _common_run_helper(env: env_tools.PreparedEnv, coverage: bool):
        pytest_path = os.path.join(env.virtual_env_path, 'bin', 'pytest')
        target_path = os.path.join(env.destination_directory, 'cirq')
        try:
            result = env_tools.sub_run(pytest_path,
                                       target_path,
                                       '--cov' if coverage else '',
                                       capture_stdout=True,
                                       raise_error_if_process_fails=False)
            print(result)
        except subprocess.CalledProcessError:
            return False, 'Tests failed.'
        return True, 'Tests passed!'


class IncrementalCoverageCheck(Check):
    def context(self):
        return 'incremental coverage by maintainer'

    def run_helper(self, env: env_tools.PreparedEnv):
        uncovered_count = incremental_coverage.check_for_uncovered_lines(env)
        if not uncovered_count:
            return True, 'All covered!'
        return False, '{} changed uncovered lines'.format(uncovered_count)


class LintCheck(Check):
    def context(self):
        return 'pylint by maintainer'

    def run_helper(self, env: env_tools.PreparedEnv):
        pylint_path = os.path.join(env.virtual_env_path, 'bin', 'pylint')
        target_path = os.path.join(env.destination_directory, 'cirq')
        try:
            result = env_tools.sub_run(
                pylint_path,
                target_path,
                '--reports=no',
                '--score=no',
                '--output-format=colorized',
                '--rcfile={}'.format(os.path.join(env.destination_directory,
                                                  'continuous-integration',
                                                  '.pylintrc')),
                capture_stdout=True,
                raise_error_if_process_fails=False)
            print(result)
        except subprocess.CalledProcessError:
            return False, 'Lint.'
        return True, 'No lint here!'


class TypeCheck(Check):
    def context(self):
        return 'typecheck by maintainer'

    def run_helper(self, env: env_tools.PreparedEnv):
        pylint_path = os.path.join(env.virtual_env_path, 'bin', 'pylint')
        target_path = os.path.join(env.destination_directory, 'cirq')
        try:
            result = env_tools.sub_run(
                pylint_path,
                target_path,
                '--reports=no',
                '--score=no',
                '--output-format=colorized',
                '--rcfile=continuous-integration/.pylintrc',
                capture_stdout=True,
                raise_error_if_process_fails=False)
            print(result)
        except subprocess.CalledProcessError:
            return False, 'Lint.'
        return True, 'No lint here!'


ALL_CHECKS = [
    LintCheck(),
    TypeCheck(),
    TestAndPrepareCoverageCheck(),
    IncrementalCoverageCheck()
]
