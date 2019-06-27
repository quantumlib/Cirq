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
import functools
from typing import Dict, Optional, List, Any, Iterable, Callable, Tuple, Union
import numpy as np
from ply import yacc

import cirq
from cirq import Circuit, NamedQubit, CX
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import._lexer import QasmLexer
from cirq.contrib.qasm_import.exception import QasmException


class Qasm:
    """Qasm stores the final result of the Qasm parsing."""

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
    """Specifies how to convert a call to an OpenQASM gate
    to a list of `cirq.GateOperation`s.

    Has the responsibility to validate the arguments
    and parameters of the call and to generate a list of corresponding
    `cirq.GateOperation`s in the `on` method.
    """

    def __init__(
            self, qasm_gate: str,
            cirq_gate: Union[cirq.Gate, Callable[[List[float]], cirq.Gate]],
            num_params: int, num_args: int):
        """Initializes a Qasm gate statement.

       Args:
           qasm_gate: the symbol of the QASM gate
           cirq_gate: the gate class on the cirq side
           num_args: the number of qubits (used in validation) this
                       gate takes
       """
        self.qasm_gate = qasm_gate
        self.cirq_gate = cirq_gate
        self.num_params = num_params
        self.num_args = num_args

    def _validate_args(self, args: List[List[cirq.Qid]], lineno: int):
        if len(args) != self.num_args:
            raise QasmException(
                "{} only takes {} arg(s) (qubits and/or registers), "
                "got: {}, at line {}".format(self.qasm_gate, self.num_args,
                                             len(args), lineno))

    def _validate_params(self, params: List[float], lineno: int):
        if len(params) != self.num_params:
            raise QasmException(
                "{} takes {} parameter(s), got: {}, at line {}".format(
                    self.qasm_gate, self.num_params, len(params), lineno))

    def on(self, params: List[float], args: List[List[cirq.Qid]],
           lineno: int) -> Iterable[cirq.Operation]:
        self._validate_args(args, lineno)
        self._validate_params(params, lineno)

        reg_sizes = np.unique([len(reg) for reg in args])
        if len(reg_sizes) > 2 or (len(reg_sizes) > 1 and reg_sizes[0] != 1):
            raise QasmException("Non matching quantum registers of length {} "
                                "at line {}".format(reg_sizes, lineno))

        # the actual gate we'll apply the arguments to might be a parameterized
        # or non-parametrized gate
        final_gate = (self.cirq_gate if isinstance(self.cirq_gate, cirq.Gate)
                      else self.cirq_gate(params))  # type: cirq.Gate
        # OpenQASM gates can be applied on single qubits and qubit registers.
        # We represent single qubits as registers of size 1.
        # Based on the OpenQASM spec (https://arxiv.org/abs/1707.03429),
        # single qubit arguments can be mixed with qubit registers.
        # Given quantum registers of length reg_size and single qubits are both
        # used as arguments, we generate reg_size GateOperations via iterating
        # through each qubit of the registers 0 to n-1 and use the same one
        # qubit from the "single-qubit registers" for each operation.
        op_qubits = functools.reduce(np.broadcast, args)  # type: np.broadcast
        for qubits in op_qubits:  # type: Tuple[cirq.Qid]
            if isinstance(qubits, cirq.Qid):
                yield final_gate.on(qubits)
            elif len(np.unique(qubits)) < len(qubits):
                raise QasmException("Overlapping qubits in arguments"
                                    " at line {}".format(lineno))
            else:
                yield final_gate.on(*qubits)


class QasmParser:
    """Parser for QASM strings.

    Example:

        qasm = "OPENQASM 2.0; qreg q1[2]; CX q1[0], q1[1];"
        parsedQasm = QasmParser().parse(qasm)
    """

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
        'CX':
        QasmGateStatement(qasm_gate='CX',
                          cirq_gate=CX,
                          num_params=0,
                          num_args=2),
        'U':
        QasmGateStatement(
            qasm_gate='U',
            num_params=3,
            num_args=1,
            # QasmUGate expects half turns and
            # changes the order of arguments
            cirq_gate=(lambda params: QasmUGate(*[p / np.pi for p in params])))
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
    #         | measurement circuit
    #         | empty

    def p_circuit_reg(self, p):
        """circuit : new_reg circuit"""
        p[0] = self.circuit

    def p_circuit_gate_or_measurement(self, p):
        """circuit : gate_op circuit
                   | measurement circuit"""
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
        if length == 0:
            raise QasmException("Illegal, zero-length register '{}' "
                                "at line {}".format(name, p.lineno(4)))
        if p[1] == "qreg":
            self.qregs[name] = length
        else:
            self.cregs[name] = length
        p[0] = (name, length)

    # gate operations
    # gate_op : ID qargs
    #         | ID () qargs
    #         | ID ( params ) qargs

    def p_gate_op_no_params(self, p):
        """gate_op :  ID qargs
                   | ID '(' ')' qargs"""
        self._resolve_gate_operation(args=p[4] if p[2] == '(' else p[2],
                                     gate=p[1],
                                     p=p,
                                     params=[])

    def p_gate_op_with_params(self, p):
        """gate_op :  ID '(' params ')' qargs"""
        self._resolve_gate_operation(args=p[5], gate=p[1], p=p, params=p[3])

    def _resolve_gate_operation(self, args: List[List[cirq.Qid]], gate: str,
                                p: Any, params: List[float]):
        if gate not in self.basic_gates.keys():
            raise QasmException('Unknown gate "{}" at line {}, '
                                'maybe you forgot to include '
                                'the standard qelib1.inc?'.format(
                                    gate, p.lineno(1)))
        p[0] = self.basic_gates[gate].on(args=args,
                                         params=params,
                                         lineno=p.lineno(1))

    # params : parameter ',' params
    #        | parameter

    def p_params_multiple(self, p):
        """params : expr ',' params"""
        p[3].insert(0, p[1])
        p[0] = p[3]

    def p_params_single(self, p):
        """params : expr """
        p[0] = [p[1]]

    # expr : term
    #            | func '(' expression ')' """
    #            | binary_op
    #            | unary_op

    def p_expr_term(self, p):
        """expr : term"""
        p[0] = p[1]

    def p_expr_parens(self, p):
        """expr : '(' expr ')'"""
        p[0] = p[2]

    def p_expr_unary(self, p):
        """expr : '-' expr
                | '+' expr """
        if p[1] == '-':
            p[0] = -p[2]
        else:
            p[0] = p[2]

    def p_term(self, p):
        """term : NUMBER
                | NATURAL_NUMBER
                | PI """
        p[0] = p[1]

    # qargs : qarg ',' qargs
    #      | qarg ';'

    def p_args_multiple(self, p):
        """qargs : qarg ',' qargs"""
        p[3].insert(0, p[1])
        p[0] = p[3]

    def p_args_single(self, p):
        """qargs : qarg ';'"""
        p[0] = [p[1]]

    # qarg : ID
    #     | ID '[' NATURAL_NUMBER ']'

    def p_quantum_arg_register(self, p):
        """qarg : ID """
        reg = p[1]
        if reg not in self.qregs.keys():
            raise QasmException('Undefined quantum register "{}" '
                                'at line {}'.format(reg, p.lineno(1)))
        qubits = []
        for idx in range(self.qregs[reg]):
            arg_name = self.make_name(idx, reg)
            if arg_name not in self.qubits.keys():
                self.qubits[arg_name] = NamedQubit(arg_name)
            qubits.append(self.qubits[arg_name])
        p[0] = qubits

    # carg : ID
    #     | ID '[' NATURAL_NUMBER ']'

    def p_classical_arg_register(self, p):
        """carg : ID """
        reg = p[1]
        if reg not in self.cregs.keys():
            raise QasmException('Undefined classical register "{}" '
                                'at line {}'.format(reg, p.lineno(1)))

        p[0] = [self.make_name(idx, reg) for idx in range(self.cregs[reg])]

    def make_name(self, idx, reg):
        return str(reg) + "_" + str(idx)

    def p_quantum_arg_bit(self, p):
        """qarg : ID '[' NATURAL_NUMBER ']' """
        reg = p[1]
        idx = p[3]
        arg_name = self.make_name(idx, reg)
        if reg not in self.qregs.keys():
            raise QasmException('Undefined quantum register "{}" '
                                'at line {}'.format(reg, p.lineno(1)))
        size = self.qregs[reg]
        if idx >= size:
            raise QasmException('Out of bounds qubit index {} '
                                'on register {} of size {} '
                                'at line {}'.format(idx, reg, size,
                                                    p.lineno(1)))
        if arg_name not in self.qubits.keys():
            self.qubits[arg_name] = NamedQubit(arg_name)
        p[0] = [self.qubits[arg_name]]

    def p_classical_arg_bit(self, p):
        """carg : ID '[' NATURAL_NUMBER ']' """
        reg = p[1]
        idx = p[3]
        arg_name = self.make_name(idx, reg)
        if reg not in self.cregs.keys():
            raise QasmException('Undefined classical register "{}" '
                                'at line {}'.format(reg, p.lineno(1)))

        size = self.cregs[reg]
        if idx >= size:
            raise QasmException('Out of bounds bit index {} '
                                'on classical register {} of size {} '
                                'at line {}'.format(idx, reg, size,
                                                    p.lineno(1)))
        p[0] = [arg_name]

    # measurement operations
    # measurement : MEASURE qarg ARROW carg

    def p_measurement(self, p):
        """measurement : MEASURE qarg ARROW carg ';'"""
        qreg = p[2]
        creg = p[4]

        if len(qreg) != len(creg):
            raise QasmException(
                'mismatched register sizes {} -> {} for measurement '
                'at line {}'.format(len(qreg), len(creg), p.lineno(1)))

        p[0] = [
            cirq.MeasurementGate(num_qubits=1, key=creg[i]).on(qreg[i])
            for i in range(len(qreg))
        ]

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
