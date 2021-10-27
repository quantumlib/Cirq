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

from astroid import parse
from pylint.testutils import CheckerTestCase, Message

from dev_tools.pylint_copyright_checker import CopyrightChecker


class TestCopyrightChecker(CheckerTestCase):
    r"""Test the copyright-notice checker for Pylint"""

    CHECKER_CLASS = CopyrightChecker

    def test_missing_copyright(self) -> None:
        r"""Report message when no copyright notice at the beginning of a file."""
        node = parse("import os")
        with self.assertAddsMessages(
            Message(
                msg_id='wrong-or-nonexistent-copyright-notice',
                line=1,
            ),
        ):
            self.checker.process_module(node)

    def test_wrong_copyright(self) -> None:
        r"""Report message when the copyright notice is incorrect."""
        node = parse("# Copyright 2021 Someone else")
        with self.assertAddsMessages(
            Message(
                msg_id='wrong-or-nonexistent-copyright-notice',
                line=1,
            ),
        ):
            self.checker.process_module(node)

    def test_shorter_copyright(self) -> None:
        r"""Report message when the copyright notice is incorrect."""
        node = parse("# Copyright 2021 The")
        with self.assertAddsMessages(
            Message(
                msg_id='wrong-or-nonexistent-copyright-notice',
                line=1,
            ),
        ):
            self.checker.process_module(node)

    def test_longer_copyright(self) -> None:
        r"""Report message when the copyright notice is incorrect."""
        node = parse("# Copyright 2021 The Cirq Developers and extra")
        with self.assertAddsMessages(
            Message(
                msg_id='wrong-or-nonexistent-copyright-notice',
                line=1,
            ),
        ):
            self.checker.process_module(node)

    def test_correct_copyright(self) -> None:
        r"""Do not report a message when the correct copyright notice is shown."""
        node = parse(
            """# Copyright 2020 The Cirq Developers
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
"""
        )
        with self.assertNoMessages():
            self.checker.process_module(node)
