import subprocess
from typing import Tuple, Optional

import abc
import os.path

from python_ci_utils import env_tools


class Check(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def context(self) -> str:
        pass

    @abc.abstractmethod
    def run_helper(self, env: env_tools.PreparedEnv) -> Tuple[bool, str]:
        pass

    def run_and_report(self, env: env_tools.PreparedEnv
                       ) -> Tuple[bool, str, Optional[Exception]]:
        env.report_status('pending', 'Running...', self.context())
        try:
            success, message = self.run_helper(env)
        except Exception as ex:
            env.report_status('error', 'Unexpected error.', self.context())
            return False, 'Unexpected error.', ex

        env.report_status(
            'success' if success else 'failure',
            message,
            self.context())
        return success, message, None


class PyTest3Check(Check):
    def context(self):
        return 'pytest by maintainer'

    def run_helper(self, env: env_tools.PreparedEnv):
        pytest_path = os.path.join(env.virtual_env_path, 'bin', 'pytest')
        target_path = os.path.join(env.destination_directory, 'cirq')
        try:
            env_tools.run_forward_into_std(pytest_path, target_path)
        except subprocess.CalledProcessError:
            return False, 'Tests failed.'
        return True, 'Tests passed!'


class PylintCheck(Check):
    def context(self):
        return 'pylint by maintainer'

    def run_helper(self, env: env_tools.PreparedEnv):
        pylint_path = os.path.join(env.virtual_env_path, 'bin', 'pylint')
        target_path = os.path.join(env.destination_directory, 'cirq')
        try:
            env_tools.run_forward_into_std(
                pylint_path,
                target_path,
                '--reports=no',
                '--score=no',
                '--output-format=colorized',
                '--rcfile=continuous-integration/.pylintrc')
        except subprocess.CalledProcessError:
            return False, 'Lint.'
        return True, 'No lint here!'


ALL_CHECKS = [PyTest3Check(), PylintCheck()]
