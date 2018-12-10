
import re

from dev_tools import check, env_tools

def RequiresRStringChange(filepath: str, verbose: bool):
    """Returns if the file contains a docstring that needs to be an r-string.
    """
    docstring_pattern = re.compile('""".*?"""', re.DOTALL)
    latex_pattern = re.compile('\\\.+?\{.+?\}')

    with open(filepath, 'r') as inputfile:
        contents = inputfile.read()
        matches = docstring_pattern.findall(contents)
        for match in matches:
            if latex_pattern.search(match):
                if verbose:
                    print('Found docstring that should be an r-string due to containing latex in: ', filepath)

                return True

    return False


class CustomLintCheck(check.Check):
    """Performs custom lint checks."""

    def command_line_switch(self):
        return "custom-lint"

    def context(self):
        return "custom-lint"

    def perform_check(self, env: env_tools.PreparedEnv, verbose: bool):
        """
        """
        passed = True

        basepath = str(env.destination_directory)
        filepaths = list(env_tools.get_unhidden_ungenerated_python_files(basepath))
        for filepath in filepaths:
            if RequiresRStringChange(filepath, verbose):
                passed = False

        if passed:
            return True, 'All custom lint checks pass'

        return False, 'Failed custom lint check'
