# Copyright 2018 The Cirq Developers
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

import re
from astroid import nodes

from pylint.checkers import BaseChecker
from pylint.interfaces import IRawChecker


class CopyrightChecker(BaseChecker):
    r"""Check for the copyright notices in the beginning of Python source files.

    This checker can be disabled by putting `# pylint: disable=wrong-copyright-notice` at the
    beginning of a file.
    """

    __implements__ = IRawChecker

    name = "copyright-notice"
    msgs = {
        "R0001": (
            "Missing or wrong copyright notice",
            "wrong-copyright-notice",
            "Consider putting a correct copyright notice at the beginning of a file.",
        )
    }
    options = ()

    def process_module(self, node: nodes.Module) -> None:
        r"""Check whether the copyright notice is correctly placed in the source file of a module.

        Compare the first lines of a source file against the standard copyright notice (i.e., the
        `golden` variable below). Suffix whitespace (including newline symbols) is not considered
        during the comparison. Pylint will report a message if the copyright notice is not
        correctly placed.

        Args:
            node: the module to be checked.
        """
        # Exit if the checker is disabled in the source file.
        if not self.linter.is_message_enabled("wrong-copyright-notice"):
            return
        golden = [
            '# Copyright \\d{4} The Cirq Developers',
            b'#',
            b'# Licensed under the Apache License, Version 2.0 (the "License");',
            b'# you may not use this file except in compliance with the License.',
            b'# You may obtain a copy of the License at',
            b'#',
            b'#     https://www.apache.org/licenses/LICENSE-2.0',
            b'#',
            b'# Unless required by applicable law or agreed to in writing, software',
            b'# distributed under the License is distributed on an "AS IS" BASIS,',
            b'# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.',
            b'# See the License for the specific language governing permissions and',
            b'# limitations under the License.',
        ]
        with node.stream() as stream:
            for (lineno, line) in enumerate(stream):
                if lineno >= len(golden):
                    break

                if lineno == 0:
                    # Use Regex to accept various years such as 2018 and 2021.
                    if not re.compile(golden[0]).match(line.decode(node.file_encoding)):
                        self.add_message("wrong-copyright-notice", line=1)
                        break
                elif line.rstrip() != golden[lineno]:
                    self.add_message("wrong-copyright-notice", line=lineno + 1)
                    break


def register(linter):
    r"""Register this checker to pylint.

    The registration is done automatically if this file is in $PYTHONPATH.
    """
    linter.register_checker(CopyrightChecker(linter))
