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
from cirq import Circuit
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer
from ply import yacc


class Qasm(object):

    def __init__(self, supportedFormat: bool, qelib1Include: bool, c: Circuit):
        # defines whether the Quantum Experience standard header
        # is present or not
        self.qelib1Include = qelib1Include
        # defines if it has a supported format or not
        self.supportedFormat = supportedFormat
        # circuit
        self.circuit = c


class QasmParser(object):

    def __init__(self, qasm: str):
        self.qasm = qasm
        self.parser = yacc.yacc(module=self,
                                debug=False,
                                write_tables=False)

    tokens = QasmLexer.tokens
    start = 'start'

    def p_start(self, p):
        """start : qasm"""
        p[0] = p[1]

    def p_qasm_0(self, p):
        """qasm : format circuit"""
        p[0] = Qasm(True, False, p[2])

    def p_qasm_1(self, p):
        """qasm : format QELIBINC circuit"""
        p[0] = Qasm(True, True, p[3])

    def p_format(self, p):
        """format : FORMAT_SPEC"""
        if p[1] != "2.0":
            raise QasmException(
                "Unsupported OpenQASM version: {}, "
                "only 2.0 is supported currently by Cirq".format(
                    p[1]), self.qasm)

    def p_qasm_error(self, p):
        """qasm : QELIBINC
                | circuit """
        raise QasmException("Missing 'OPENQASM 2.0;' statement", self.qasm)

    def p_circuit(self, p):
        """circuit : empty"""
        p[0] = Circuit()

    def p_error(self, p):
        raise QasmException("Syntax error in input on {}".format(p), self.qasm)

    def p_empty(self, p):
        """empty :"""
        pass

    def parse(self) -> Qasm:
        return self.parser.parse(lexer=QasmLexer(self.qasm))
