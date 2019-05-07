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
import operator
from typing import List, Any, Union, Iterable, Callable, Dict, Optional

import sympy
from ply import yacc

import cirq
from cirq import Circuit, NamedQubit, CNOT
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import._lexer import QasmLexer
from cirq.contrib.qasm_import.exception import QasmException


class Qasm(object):

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


class QasmParser(object):

    def validate_params(self, qasm_gate, params, num_params, lineno):
        if len(params) != num_params:
            raise QasmException(
                "{} takes {} parameter(s), got: {}, at line {}".format(
                    qasm_gate, num_params, len(params), lineno))

    def validate_args(self, qasm_gate, args, num_args, lineno):
        if len(args) != num_args:
            raise QasmException(
                "{} only takes {} arg(s) (qubits and/or registers), "
                "got: {}, at line {}".format(qasm_gate, num_args, len(args),
                                             lineno))

    def make_gate(self, qasm_gate: str, cirq_gate: Callable[[Any], cirq.Gate],
                  num_params: int, num_args: int):

        def call_gate(params: List[sympy.Number],
                      args: List[List[cirq.Qid]],
                      lineno: int = 0) -> Iterable[cirq.GateOperation]:
            self.validate_params(qasm_gate, params, num_params, lineno)
            self.validate_args(qasm_gate, args, num_args, lineno)
            reg_size = 1
            for reg in args:
                if len(reg) > 1 and len(reg) != reg_size:
                    if reg_size == 1:
                        reg_size = len(reg)
                    else:
                        raise QasmException(
                            "Non matching quantum registers of length {} "
                            "at line {}".format([len(reg) for reg in args],
                                                lineno))

            for qbit_index in range(reg_size):
                final_gate = cirq_gate if isinstance(cirq_gate, cirq.Gate) \
                    else cirq_gate(*[float(p)
                                     for p in params])  # type: cirq.Gate

                yield final_gate.on(
                    *[qreg[min(len(qreg) - 1, qbit_index)] for qreg in args])

        return call_gate

    def u_gate(self):
        operation = self.make_gate('U', QasmUGate, num_args=1, num_params=3)

        def call_gate(params: List[sympy.Number],
                      args: List[List[cirq.Qid]],
                      lineno: int = 0) -> Iterable[cirq.GateOperation]:
            self.validate_params('U', params, 3, lineno)
            return operation([params[2], params[0], params[1]], args, lineno)

        return call_gate

    def id_gate(self,
                params: List[sympy.Number],
                args: List[List[cirq.Qid]],
                lineno: int = 0) -> Iterable[cirq.GateOperation]:
        self.validate_args('id', args, 1, lineno)
        self.validate_params('id', params, 0, lineno)
        return cirq.IdentityGate(len(args[0]))(*args[0])

    def __init__(self, qasm: str):
        self.basic_gates = {
            'CX': self.make_gate('CX', CNOT, num_params=0, num_args=2),
            'U': self.u_gate()
        }
        self.standard_gates = {
            'u3':
            self.u_gate(),
            'x':
            self.make_gate('x', cirq.X, num_params=0, num_args=1),
            'y':
            self.make_gate('y', cirq.Y, num_params=0, num_args=1),
            'z':
            self.make_gate('z', cirq.Z, num_params=0, num_args=1),
            'h':
            self.make_gate('h', cirq.H, num_params=0, num_args=1),
            's':
            self.make_gate('s', cirq.S, num_params=0, num_args=1),
            't':
            self.make_gate('t', cirq.T, num_params=0, num_args=1),
            'sdg':
            self.make_gate('sdg', cirq.S**-1, num_params=0, num_args=1),
            'tdg':
            self.make_gate('tdg', cirq.T**-1, num_params=0, num_args=1),
            'cx':
            self.make_gate('cx', CNOT, num_params=0, num_args=2),
            'cy':
            self.make_gate('cy',
                           cirq.ControlledGate(cirq.Y),
                           num_params=0,
                           num_args=2),
            'cz':
            self.make_gate('cz', cirq.CZ, num_params=0, num_args=2),
            'swap':
            self.make_gate('swap', cirq.SWAP, num_params=0, num_args=2),
            'ch':
            self.make_gate('ch',
                           cirq.ControlledGate(cirq.H),
                           num_params=0,
                           num_args=2),
            'id':
            self.id_gate,
            'rx':
            self.make_gate('rx', cirq.Rx, num_params=1, num_args=1),
            'ry':
            self.make_gate('ry', cirq.Ry, num_params=1, num_args=1),
            'rz':
            self.make_gate('rz', cirq.Rz, num_params=1, num_args=1),
            'ccx':
            self.make_gate('ccx', cirq.TOFFOLI, num_args=3, num_params=0),
            'cswap':
            self.make_gate('cswap', cirq.CSWAP, num_args=3, num_params=0)
        }
        self.standard_gates.update(**self.basic_gates)

        self.qasm = qasm
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
        self.circuit = Circuit()
        self.qregs = {}  # type: Dict[str,int]
        self.cregs = {}  # type: Dict[str,int]
        self.qelibinc = False
        self.qubits = {}  # type: Dict[str,cirq.NamedQubit]
        self.functions = {
            'sin': sympy.sin,
            'cos': sympy.cos,
            'tan': sympy.tan,
            'exp': sympy.exp,
            'ln': sympy.ln,
            'sqrt': sympy.sqrt,
            'acos': sympy.acos,
            'atan': sympy.atan,
            'asin': sympy.asin
        }

        self.binary_operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '^': operator.pow
        }
        self.parsedQasm = None  # type: Optional[Qasm]
        self.supported_format = False

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
        if length <= 0:
            raise QasmException("Illegal, zero-length register '{}' "
                                "at line {}".format(name, p.lineno(4)))
        if p[1] == "qreg":
            self.qregs[name] = length
        else:
            self.cregs[name] = length
        p[0] = (name, length)

    # gate operations
    # gate_op : ID args
    #         | ID () args
    #         | ID ( params ) args

    def p_gate_op_no_params(self, p):
        """gate_op :  ID args
                   | ID '(' ')' args"""
        self._resolve_gate_operation(args=p[4] if p[2] == '(' else p[2],
                                     gate=p[1],
                                     p=p,
                                     params=[])

    def p_gate_op_with_params(self, p):
        """gate_op :  ID '(' params ')' args"""
        self._resolve_gate_operation(args=p[5], gate=p[1], p=p, params=p[3])

    def _resolve_gate_operation(self, args: List[List[cirq.Qid]], gate: str,
                                p: Any, params: List[sympy.Number]):
        if self.qelibinc is False:
            if gate not in self.basic_gates.keys():
                raise QasmException('Unknown gate "{}" at line {}, '
                                    'maybe you forgot to include '
                                    'the standard qelib1.inc?'.format(
                                        gate, p.lineno(1)))
            p[0] = self.basic_gates[gate](args=args,
                                          params=params,
                                          lineno=p.lineno(1))
        else:
            if gate not in self.standard_gates.keys():
                raise QasmException('Unknown gate "{}" at line {}'.format(
                    gate, p.lineno(1)))

            p[0] = self.standard_gates[gate](args=args,
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

    def p_expr_function_call(self, p):
        """expr : ID '(' expr ')'"""
        func = p[1]
        if func not in self.functions.keys():
            raise QasmException(
                "Function not recognized: '{}' at line {}".format(
                    func, p.lineno(1)))
        p[0] = self.functions[func](p[3])

    def p_expr_unary(self, p):
        """expr : '-' expr
                | '+' expr """
        if p[1] == '-':
            p[0] = -p[2]
        else:
            p[0] = p[2]

    def p_expr_binary(self, p):
        """expr : expr '*' expr
                | expr '/' expr
                | expr '+' expr
                | expr '-' expr
                | expr '^' expr
        """
        p[0] = self.binary_operators[p[2]](p[1], p[3])

    def p_term(self, p):
        """term : NUMBER
                | NATURAL_NUMBER
                | PI """
        p[0] = p[1]

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
        elif reg in self.cregs.keys():
            keys = []
            for num in range(self.cregs[reg]):
                arg_name = self.make_name(num, reg)
                keys.append(arg_name)
            p[0] = keys
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
            if arg_name not in self.qubits.keys():
                self.qubits[arg_name] = NamedQubit(arg_name)
            p[0] = [self.qubits[arg_name]]
        elif reg in self.cregs.keys():
            p[0] = [arg_name]
        else:
            raise QasmException('Undefined quantum/classical register "{}" '
                                'at line {}'.format(reg, p.lineno(1)))

    # measurement operations
    # measurement : MEASURE arg ARROW arg
    def p_measurement(self, p):
        """measurement : MEASURE arg ARROW arg ';'"""
        qreg = p[2]
        creg = p[4]

        if len(qreg) != len(creg):
            raise QasmException(
                'mismatched register sizes {} -> {} for measurement '
                'at line {}'.format(len(qreg), len(creg), p.lineno(1)))

        measurements = []
        for i in range(len(qreg)):
            measurements.append(
                cirq.MeasurementGate(num_qubits=1, key=creg[i]).on(qreg[i]))
        p[0] = measurements

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

    def parse(self) -> Qasm:
        if self.parsedQasm is None:
            self.parsedQasm = self.parser.parse(lexer=QasmLexer(self.qasm))
        return self.parsedQasm

    def debug_context(self, p):
        debug_start = max(self.qasm.rfind('\n', 0, p.lexpos) + 1, p.lexpos - 5)
        debug_end = min(self.qasm.find('\n', p.lexpos, p.lexpos + 5),
                        p.lexpos + 5)

        return "..." + self.qasm[debug_start:debug_end] + "\n" + (
            " " * (3 + p.lexpos - debug_start)) + "^"
