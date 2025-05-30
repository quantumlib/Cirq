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

from __future__ import annotations

import re

import ply.lex as lex

from cirq.contrib.qasm_import.exception import QasmException


class QasmLexer:
    def __init__(self):
        self.lex = lex.lex(object=self, debug=False)

    literals = "{}[]();,+/*-^="

    reserved = {
        'qubit': 'QUBIT',
        'qreg': 'QREG',
        'bit': 'BIT',
        'creg': 'CREG',
        'measure': 'MEASURE',
        'reset': 'RESET',
        'gate': 'GATE',
        'if': 'IF',
        'pi': 'PI',
    }

    tokens = [
        'FORMAT_SPEC',
        'NUMBER',
        'NATURAL_NUMBER',
        'STDGATESINC',
        'QELIBINC',
        'ID',
        'ARROW',
        'EQ',
    ] + list(reserved.values())

    def t_newline(self, t):
        r"""\n+"""
        t.lexer.lineno += len(t.value)

    t_ignore = ' \t'

    # all numbers except NATURAL_NUMBERs:
    # it's useful to have this separation to be able to handle indices
    # separately. In case of the parameter expressions, we are "OR"-ing
    # them together (see p_term in _parser.py)
    def t_NUMBER(self, t):
        r"""(
        (
        [0-9]+\.?|
        [0-9]?\.[0-9]+
        )
        [eE][+-]?[0-9]+
        )|
        (
        ([0-9]+)?\.[0-9]+|
        [0-9]+\.)"""
        t.value = float(t.value)
        return t

    def t_NATURAL_NUMBER(self, t):
        r"""\d+"""
        t.value = int(t.value)
        return t

    def t_FORMAT_SPEC(self, t):
        r"""OPENQASM(\s+)([^\s\t\;]*);"""
        match = re.match(r"""OPENQASM(\s+)([^\s\t;]*);""", t.value)
        t.value = match.groups()[1]
        return t

    def t_QELIBINC(self, t):
        r"""include(\s+)"qelib1.inc";"""
        return t

    def t_STDGATESINC(self, t):
        r"""include(\s+)"stdgates.inc";"""
        return t

    def t_ARROW(self, t):
        """->"""
        return t

    def t_EQ(self, t):
        """=="""
        return t

    def t_ID(self, t):
        r"""[a-zA-Z][a-zA-Z\d_]*"""
        if t.value in QasmLexer.reserved:
            t.type = QasmLexer.reserved[t.value]
        return t

    def t_COMMENT(self, t):
        r"""//.*"""

    def t_error(self, t):
        raise QasmException(f"Illegal character '{t.value[0]}' at line {t.lineno}")

    def input(self, qasm):
        self.lex.input(qasm)

    def token(self) -> lex.Token | None:
        return self.lex.token()
