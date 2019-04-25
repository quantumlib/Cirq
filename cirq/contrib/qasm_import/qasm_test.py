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

from cirq.contrib.qasm_import import QasmCircuitParser, QasmException


def test_missing_header():
    assert_parse_error("", r"""missing QASM header""")


def assert_parse_error(qasm_string: str, error_regexp):
    try:
        circuit = QasmCircuitParser(qasm_string).parse()
        raise AssertionError(
            'Expected QASM parser to throw error matching `{}`,'
            ' instead returned {}'.format(error_regexp, circuit))
    except QasmException as ex:
        assert re.match(error_regexp, ex.message)
        assert qasm_string == ex.qasm
