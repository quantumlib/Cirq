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
from pylint.testutils import CheckerTestCase, MessageTest

from dev_tools.import-only-modules import ImportOnlyModulesChecker


class TestImportChecker(CheckerTestCase):
    r"""Test the imports only modules checker for Pylint"""

    CHECKER_CLASS = ImportOnlyModulesChecker

    def test_wrong_import(self) -> None:
        r"""Report message when no copyright notice at the beginning of a file."""
        node = parse("from cirq.devices import GridQubit")
        with self.assertAddsMessages(
            MessageTest(msg_id='import-only-modules', col_offset=0)
        ):
            self.checker.process_module(node)

    def test_correct_import(self) -> None:
        r"""Report message when no copyright notice at the beginning of a file."""
        node = parse("from cirq import devices")
        with self.assertNoMessages():
            self.checker.process_module(node)

    def test_correct_import(self) -> None:
        r"""Report message when no copyright notice at the beginning of a file."""
        node = parse("import cirq")
        with self.assertNoMessages():
            self.checker.process_module(node)

