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
from typing import Optional

import ply.lex as lex
import sympy
from sympy import Number

from cirq.contrib.qasm_import.exception import QasmException


class QasmLexer(object):
    NATURAL_NUMBER = "NATURAL_NUMBER"

    def __init__(self, qasm: str):
        self.qasm = qasm
        self.lex = lex.lex(object=self, debug=False)
        self.lex.input(qasm)

    literals = "{}[]();,+/*-^"

    reserved = {
        'qreg': 'QREG',
        'creg': 'CREG',
        'measure': 'MEASURE',
        '->': 'ARROW',
    }

    tokens = [
                 'FORMAT_SPEC',
                 'NUMBER',
                 'NATURAL_NUMBER',
                 'QELIBINC',
                 'ID',
                 'PI',
             ] + list(reserved.values())

    def t_newline(self, t):
        r"""\n+"""
        t.lexer.lineno += len(t.value)

    t_ignore = ' \t'

    def t_PI(selfs, t):
        r"""pi"""
        t.value = sympy.pi
        return t

    def t_NUMBER(selfs, t):
        # pylint: disable=line-too-long
        r"""(([0-9]+|([0-9]+)?\.[0-9]+|[0-9]+\.)[eE][+-]?[0-9]+)|(([0-9]+)?\.[0-9]+|[0-9]+\.)"""
        t.value = Number(t.value)
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

    def t_QREG(self, t):
        r"""qreg"""
        return t

    def t_CREG(self, t):
        r"""creg"""
        return t

    def t_MEASURE(self, t):
        r"""measure"""
        return t

    def t_ARROW(self, t):
        """->"""
        return t

    def t_ID(self, t):
        r"""[a-zA-Z][a-zA-Z\d_]*"""
        return t

    def t_COMMENT(self, t):
        r"""//.*"""

    def t_error(self, t):
        raise QasmException("Illegal character '{}' at line {}"
                            .format(t.value[0], t.lineno))

    def token(self) -> Optional[lex.Token]:
        return self.lex.token()
