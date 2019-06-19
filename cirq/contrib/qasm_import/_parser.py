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
from typing import Dict, Optional, List, Any, Iterable

from ply import yacc

import cirq
from cirq import Circuit, NamedQubit, CX
from cirq.contrib.qasm_import._lexer import QasmLexer
from cirq.contrib.qasm_import.exception import QasmException


class Qasm:

    def __init__(self, supported_format: bool, qelib1_include: bool,
                 qregs: dict, cregs: dict, c: Circuit):
        # defines whether the Quantum Experience standard header
        # is present or not
        self.qelib1Include = qelib1_include
        # defines if it has a supported format or not
        self.supportedFormat = supported_format
        # circuit
        self.qregs = qregs
        self.cregs = cregs
        self.circuit = c


class QasmGateStatement:

    def __init__(self, qasm_gate: str, cirq_gate: cirq.Gate, num_args: int):
        self.qasm_gate = qasm_gate
        self.cirq_gate = cirq_gate
        self.num_args = num_args

    def validate_args(self, args: List[List[cirq.Qid]], lineno: int):
        if len(args) != self.num_args:
            raise QasmException(
                "{} only takes {} arg(s) (qubits and/or registers), "
                "got: {}, at line {}".format(self.qasm_gate, self.num_args,
                                             len(args), lineno))

    def on(self, args: List[List[cirq.Qid]],
           lineno: int) -> Iterable[cirq.Operation]:
        self.validate_args(args, lineno)
        reg_size = 1
        for reg in args:
            if len(reg) > 1 and len(reg) != reg_size:
                if reg_size == 1:
                    reg_size = len(reg)
                else:
                    raise QasmException(
                        "Non matching quantum registers of length {} "
                        "at line {}".format([len(reg) for reg in args], lineno))

        # OpenQASM gates can be applied on single qubits and qubit registers.
        # We represent single qubits as registers of size 1.
        # Based on the OpenQASM spec (https://arxiv.org/abs/1707.03429),
        # single qubit arguments can be mixed with qubit registers.
        # Given quantum registers of length reg_size and single qubits are both
        # used as arguments, we generate reg_size GateOperations via iterating
        # through each qubit of the registers 0 to n-1 and use the same one
        # qubit from the "single-qubit registers" for each operation.
        for i in range(reg_size):
            qubits = []  # type: List[cirq.Qid]
            for qreg in args:
                if len(qreg) == 1:  # single qubits
                    qubit = qreg[0]
                else:  # reg_size size register
                    qubit = qreg[i]
                if qubit in qubits:
                    raise QasmException("Overlapping qubits in arguments"
                                        " at line {}".format(lineno))
                qubits.append(qubit)
            yield self.cirq_gate.on(*qubits)


class QasmParser:

    def __init__(self):
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
        self.circuit = Circuit()
        self.qregs = {}  # type: Dict[str,int]
        self.cregs = {}  # type: Dict[str,int]
        self.qelibinc = False
        self.lexer = QasmLexer()
        self.supported_format = False
        self.parsedQasm = None  # type: Optional[Qasm]
        self.qubits = {}  # type: Dict[str,cirq.NamedQubit]

    basic_gates = {
        'CX': QasmGateStatement(qasm_gate='CX', cirq_gate=CX, num_args=2)
    }  # type: Dict[str, QasmGateStatement]

    tokens = QasmLexer.tokens
    start = 'start'

    def p_start(self, p):
        """start : qasm"""
        p[0] = p[1]

    def p_qasm_format_only(self, p):
        """qasm : format"""
        self.supported_format = True
        p[0] = Qasm(self.supported_format, self.qelibinc, self.qregs,
                    self.cregs, self.circuit)

    def p_qasm_no_format_specified_error(self, p):
        """qasm : QELIBINC
                | circuit """
        if self.supported_format is False:
            raise QasmException("Missing 'OPENQASM 2.0;' statement")

    def p_qasm_include(self, p):
        """qasm : qasm QELIBINC"""
        self.qelibinc = True
        p[0] = Qasm(self.supported_format, self.qelibinc, self.qregs,
                    self.cregs, self.circuit)

    def p_qasm_circuit(self, p):
        """qasm : qasm circuit"""
        p[0] = Qasm(self.supported_format, self.qelibinc, self.qregs,
                    self.cregs, p[2])

    def p_format(self, p):
        """format : FORMAT_SPEC"""
        if p[1] != "2.0":
            raise QasmException(
                "Unsupported OpenQASM version: {}, "
                "only 2.0 is supported currently by Cirq".format(p[1]))

    # circuit : new_reg circuit
    #         | gate_op circuit
    #         | empty

    def p_circuit_reg(self, p):
        """circuit : new_reg circuit"""
        p[0] = self.circuit

    def p_circuit_gate(self, p):
        """circuit : gate_op circuit"""
        self.circuit.insert(0, p[1])
        p[0] = self.circuit

    def p_circuit_empty(self, p):
        """circuit : empty"""
        p[0] = self.circuit

    # qreg and creg

    def p_new_reg(self, p):
        """new_reg : QREG ID '[' NATURAL_NUMBER ']' ';'
                    | CREG ID '[' NATURAL_NUMBER ']' ';'"""
        name, length = p[2], p[4]
        if name in self.qregs.keys() or name in self.cregs.keys():
            raise QasmException("{} is already defined "
                                "at line {}".format(name, p.lineno(2)))
        if p[1] == "qreg":
            self.qregs[name] = length
        else:
            self.cregs[name] = length
        p[0] = (name, length)

        # gate operations
        # gate_op : ID args

    def p_gate_op_no_params(self, p):
        """gate_op :  ID args"""
        self._resolve_gate_operation(args=p[2], gate=p[1], p=p)

    def _resolve_gate_operation(self, args: List[List[cirq.Qid]], gate: str,
                                p: Any):
        if gate not in self.basic_gates.keys():
            raise QasmException('Unknown gate "{}" at line {}, '
                                'maybe you forgot to include '
                                'the standard qelib1.inc?'.format(
                                    gate, p.lineno(1)))
        p[0] = self.basic_gates[gate].on(args=args, lineno=p.lineno(1))

        # args : arg ',' args
        #      | arg ';'

    def p_args_multiple(self, p):
        """args : arg ',' args"""
        p[3].insert(0, p[1])
        p[0] = p[3]

    def p_args_single(self, p):
        """args : arg ';'"""
        p[0] = [p[1]]

        # arg : ID
        #     | ID '[' NATURAL_NUMBER ']'
        #

    def p_arg_register(self, p):
        """arg : ID """
        reg = p[1]
        if reg in self.qregs.keys():
            qubits = []
            for num in range(self.qregs[reg]):
                arg_name = self.make_name(num, reg)
                if arg_name not in self.qubits.keys():
                    self.qubits[arg_name] = NamedQubit(arg_name)
                qubits.append(self.qubits[arg_name])
            p[0] = qubits
        else:
            raise QasmException('Undefined quantum/classical register "{}" '
                                'at line {}'.format(reg, p.lineno(1)))

    def make_name(self, num, reg):
        return str(reg) + "_" + str(num)

    def p_arg_bit(self, p):
        """arg : ID '[' NATURAL_NUMBER ']' """
        reg = p[1]
        num = p[3]
        arg_name = self.make_name(num, reg)
        if reg in self.qregs.keys():
            size = self.qregs[reg]
            if num > size - 1:
                raise QasmException('Out of bounds qubit index {} '
                                    'on register {} of size {} '
                                    'at line {}'.format(num, reg, size,
                                                        p.lineno(1)))
            if arg_name not in self.qubits.keys():
                self.qubits[arg_name] = NamedQubit(arg_name)
            p[0] = [self.qubits[arg_name]]
        else:
            raise QasmException('Undefined quantum register "{}" '
                                'at line {}'.format(reg, p.lineno(1)))

    def p_error(self, p):
        if p is None:
            raise QasmException('Unexpected end of file')

        raise QasmException("""Syntax error: '{}'
{}
at line {}, column {}""".format(p.value, self.debug_context(p), p.lineno,
                                self.find_column(p)))

    def find_column(self, p):
        line_start = self.qasm.rfind('\n', 0, p.lexpos) + 1
        return (p.lexpos - line_start) + 1

    def p_empty(self, p):
        """empty :"""

    def parse(self, qasm: str) -> Qasm:
        if self.parsedQasm is None:
            self.qasm = qasm
            self.lexer.input(self.qasm)
            self.parsedQasm = self.parser.parse(lexer=self.lexer)
        return self.parsedQasm

    def debug_context(self, p):
        debug_start = max(self.qasm.rfind('\n', 0, p.lexpos) + 1, p.lexpos - 5)
        debug_end = min(self.qasm.find('\n', p.lexpos, p.lexpos + 5),
                        p.lexpos + 5)

        return "..." + self.qasm[debug_start:debug_end] + "\n" + (
            " " * (3 + p.lexpos - debug_start)) + "^"
