# Copyright 2021 The Cirq Developers
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

from astroid import nodes

from pylint.checkers import BaseChecker
from pylint.interfaces import IRawChecker


class CopyrightChecker(BaseChecker):
    r"""Check for the copyright notices at the beginning of a Python source file.

    This checker can be disabled by putting `# pylint: disable=wrong-or-nonexistent-copyright-notice`
    at the beginning of a file.
    """

    __implements__ = IRawChecker

    # The priority must be negtive. Pylint runs plugins with smaller priorities first.
    priority = -1

    name = "copyright-notice"
    msgs = {
        "R0001": (
            "Missing or wrong copyright notice",
            "wrong-or-nonexistent-copyright-notice",
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
        if not self.linter.is_message_enabled("wrong-or-nonexistent-copyright-notice"):
            return
        golden = [
            b'# Copyright 20XX The Cirq Developers',
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

            def skip_shebang(stream):
                """Skip shebang line if present, and blank lines between shebang and content."""
                lines = iter(enumerate(stream))
                try:
                    lineno, line = next(lines)
                except StopIteration:
                    return
                if line.startswith(b'#!'):
                    # skip shebang and blank lines
                    for lineno, line in lines:
                        if line.strip():
                            yield lineno, line
                            break
                else:
                    # no shebang
                    yield lineno, line
                yield from lines

            for expected_line, (i, (lineno, line)) in zip(golden, enumerate(skip_shebang(stream))):
                for expected_char, (colno, char) in zip(expected_line, enumerate(line)):
                    # The text needs to be same as the template except for the year.
                    if expected_char != char and not (i == 0 and 14 <= colno <= 15):
                        self.add_message(
                            "wrong-or-nonexistent-copyright-notice",
                            line=lineno + 1,
                            col_offset=colno,
                        )
                        return
                # The line cannot be shorter than the template or contain extra text.
                if len(line) < len(expected_line) or line[len(expected_line) :].strip() != b'':
                    self.add_message(
                        "wrong-or-nonexistent-copyright-notice",
                        line=lineno + 1,
                        col_offset=min(len(line), len(expected_line)),
                    )
                    return


def register(linter):
    r"""Register this checker to pylint.

    The registration is done automatically if this file is in $PYTHONPATH.
    """
    linter.register_checker(CopyrightChecker(linter))
