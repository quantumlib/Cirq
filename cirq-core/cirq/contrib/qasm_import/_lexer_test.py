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

import pytest
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer


def test_empty_circuit():
    lexer = QasmLexer()
    lexer.input("")
    assert lexer.token() is None


@pytest.mark.parametrize('number', ["00000", "03", "3", "0045", "21"])
def test_natural_numbers(number: str):
    lexer = QasmLexer()
    lexer.input(number)
    token = lexer.token()
    assert token is not None
    assert token.type == "NATURAL_NUMBER"
    assert token.value == int(number)


def test_supported_format():
    lexer = QasmLexer()
    lexer.input("OPENQASM 2.0;")
    token = lexer.token()
    assert token is not None
    assert token.type == "FORMAT_SPEC"
    assert token.value == '2.0'


def test_qelib_inc():
    lexer = QasmLexer()
    lexer.input('include "qelib1.inc";')
    token = lexer.token()
    assert token is not None
    assert token.type == "QELIBINC"
    assert token.value == 'include "qelib1.inc";'


def test_measurement():
    lexer = QasmLexer()
    lexer.input("measure q -> c;")
    token = lexer.token()
    assert token.type == "MEASURE"
    assert token.value == 'measure'

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == 'q'

    token = lexer.token()
    assert token.type == "ARROW"
    assert token.value == '->'

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == 'c'

    token = lexer.token()
    assert token.type == ";"
    assert token.value == ';'


@pytest.mark.parametrize(
    'identifier', ['b', 'CX', 'abc', 'aXY03', 'a_valid_name_with_02_digits_and_underscores']
)
def test_valid_ids(identifier: str):
    lexer = QasmLexer()
    lexer.input(identifier)
    token = lexer.token()

    assert token is not None
    assert token.type == "ID"
    assert token.value == identifier


@pytest.mark.parametrize(
    'number',
    ['1e2', '1e0', '3.', '4.e10', '.333', '1.0', '0.1', '2.0e-05', '1.2E+05', '123123.2132312'],
)
def test_numbers(number: str):
    lexer = QasmLexer()
    lexer.input(number)
    token = lexer.token()

    assert token is not None
    assert token.type == "NUMBER"
    assert token.value == float(number)


@pytest.mark.parametrize('token', QasmLexer.reserved.keys())
def test_keywords(token):
    lexer = QasmLexer()
    identifier = f'{token} {token}'
    lexer.input(identifier)
    t = lexer.token()
    assert t.type == QasmLexer.reserved[token]
    assert t.value == token
    t2 = lexer.token()
    assert t2.type == QasmLexer.reserved[token]
    assert t2.value == token


@pytest.mark.parametrize('token', QasmLexer.reserved.keys())
@pytest.mark.parametrize('separator', ['', '_'])
def test_identifier_starts_or_ends_with_keyword(token, separator):
    lexer = QasmLexer()
    identifier = f'{token}{separator}{token}'
    lexer.input(identifier)
    t = lexer.token()
    assert t.type == "ID"
    assert t.value == identifier


def test_qreg():
    lexer = QasmLexer()
    lexer.input('qreg [5];')
    token = lexer.token()
    assert token.type == "QREG"
    assert token.value == "qreg"

    token = lexer.token()
    assert token.type == "["
    assert token.value == "["

    token = lexer.token()
    assert token.type == "NATURAL_NUMBER"
    assert token.value == 5

    token = lexer.token()
    assert token.type == "]"
    assert token.value == "]"

    token = lexer.token()
    assert token.type == ";"
    assert token.value == ";"


def test_creg():
    lexer = QasmLexer()
    lexer.input('creg [8];')
    token = lexer.token()
    assert token.type == "CREG"
    assert token.value == "creg"

    token = lexer.token()
    assert token.type == "["
    assert token.value == "["

    token = lexer.token()
    assert token.type == "NATURAL_NUMBER"
    assert token.value == 8

    token = lexer.token()
    assert token.type == "]"
    assert token.value == "]"

    token = lexer.token()
    assert token.type == ";"
    assert token.value == ";"


def test_custom_gate():
    lexer = QasmLexer()
    lexer.input('gate name(param1,param2) q1, q2 {X(q1)}')
    token = lexer.token()
    assert token.type == "GATE"
    assert token.value == "gate"

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == "name"

    token = lexer.token()
    assert token.type == "("
    assert token.value == "("

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == "param1"

    token = lexer.token()
    assert token.type == ","
    assert token.value == ","

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == "param2"

    token = lexer.token()
    assert token.type == ")"
    assert token.value == ")"

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == "q1"

    token = lexer.token()
    assert token.type == ","
    assert token.value == ","

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == "q2"

    token = lexer.token()
    assert token.type == "{"
    assert token.value == "{"

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == "X"

    token = lexer.token()
    assert token.type == "("
    assert token.value == "("

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == "q1"

    token = lexer.token()
    assert token.type == ")"
    assert token.value == ")"

    token = lexer.token()
    assert token.type == "}"
    assert token.value == "}"


def test_error():
    lexer = QasmLexer()
    lexer.input('θ')

    with pytest.raises(QasmException, match="Illegal character 'θ' at line 1"):
        lexer.token()
