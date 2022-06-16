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

import astroid
from pylint import testutils

from dev_tools import import_only_modules


class TestImportChecker(testutils.CheckerTestCase):
    r"""Test the imports only modules checker for Pylint"""

    CHECKER_CLASS = import_only_modules.ImportOnlyModulesChecker

    def test_wrong_import(self) -> None:
        r"""Report a message when a non-module is imported"""
        node = astroid.extract_node("from cirq.devices import GridQubit")
        with self.assertAddsMessages(
            testutils.MessageTest(
                msg_id='import-only-modules',
                node=node,
                line=1,
                col_offset=0,
                end_line=1,
                end_col_offset=34,
                args=('GridQubit', 'cirq.devices')
            )
        ):
            self.checker.visit_importfrom(node)

    def test_correct_import(self) -> None:
        r"""Report no messages when correct import is used."""
        node = astroid.extract_node("from cirq import devices")
        with self.assertNoMessages():
            self.checker.visit_importfrom(node)
