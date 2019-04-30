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

from ply import yacc

from cirq import Circuit, NamedQubit, CNOT
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer


class Qasm(object):

    def __init__(self, supportedFormat: bool, qelib1Include: bool, qregs: dict,
                 cregs: dict,
                 c: Circuit):
        # defines whether the Quantum Experience standard header
        # is present or not
        self.qelib1Include = qelib1Include
        # defines if it has a supported format or not
        self.supportedFormat = supportedFormat
        # circuit
        self.qregs = qregs
        self.cregs = cregs
        self.circuit = c


class QasmParser(object):

    def __init__(self, qasm: str):
        self.qasm = qasm
        self.parser = yacc.yacc(module=self,
                                debug=False,
                                write_tables=False)
        self.circuit = Circuit()
        self.qregs = {}
        self.cregs = {}
        self.parsedQasm = None
        self.qubits = {}
        self.external_functions = ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt',
                                   'acos', 'atan', 'asin']

        self.binary_operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '^': operator.pow
        }

    tokens = QasmLexer.tokens
    start = 'start'

    def p_start(self, p):
        """start : qasm"""
        p[0] = p[1]

    def p_qasm_0(self, p):
        """qasm : format circuit"""
        p[0] = Qasm(True, False, self.qregs, self.cregs, p[2])

    def p_qasm_1(self, p):
        """qasm : format QELIBINC circuit"""
        p[0] = Qasm(True, True, self.qregs, self.cregs, p[3])

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

    # circuit : new_qreg circuit
    #         | new_creg circuit
    #         | gate_op circuit
    #         | empty

    def p_circuit_qreg(self, p):
        """circuit : new_qreg circuit"""
        p[0] = self.circuit

    def p_circuit_creg(self, p):
        """circuit : new_creg circuit"""
        name, length = p[1]
        self.cregs[name] = length
        p[0] = self.circuit

    def p_circuit_gate_op(self, p):
        """circuit : gate_op circuit"""
        self.circuit.insert(0, p[1])
        p[0] = self.circuit

    def p_circuit_empty(self, p):
        """circuit : empty"""
        p[0] = self.circuit

    # qreg and creg

    def p_new_qreg_0(self, p):
        """new_qreg : QREG ID '[' NATURAL_NUMBER ']' ';' """
        name, length = p[2], p[4]
        self.qregs[name] = length
        p[0] = (name, length)

    def p_new_creg_0(self, p):
        """new_creg : CREG ID '[' NATURAL_NUMBER ']' ';' """
        p[0] = (p[2], p[4])

    # gate operations
    # gate_op : ID args
    #         | ID ( params ) args

    def p_gate_op_no_params(self, p):
        """gate_op :  ID args
                   | ID '(' ')' args"""

        if p[2] == '(':
            args = p[4]
        else:
            args = p[2]

        if p[1] == "CX":
            if len(args) != 2:
                raise QasmException(
                    "CX only takes 2 args, got: {}, at line {}"
                        .format(len(args), p.lineno(2)),
                    self.qasm)
            ctrl_register = args[0]
            target_register = args[1]

            if len(ctrl_register) == 1 and len(target_register) == 1:
                p[0] = CNOT(ctrl_register[0], target_register[0])
            elif len(ctrl_register) == 1 and len(target_register) > 1:
                p[0] = [CNOT(ctrl_register[0], target_qubit)
                        for target_qubit in target_register]
            elif len(ctrl_register) > 1 and len(target_register) == 1:
                p[0] = [CNOT(ctrl_qubit, target_register[0])
                        for ctrl_qubit in ctrl_register]
            elif len(ctrl_register) == len(target_register):
                p[0] = [CNOT(ctrl_register[i], target_register[i])
                        for i in range(len(ctrl_register))]
            else:
                raise QasmException(
                    "Non matching quantum registers of length {} and {} "
                    "at line {}".format(
                        len(ctrl_register), len(target_register), p.lineno(1)),
                    self.qasm)

    def p_gate_op_with_params(self, p):
        """gate_op :  ID '(' params ')' args"""
        id = p[1]
        params = p[3]
        args = p[5]
        if id == "U":
            if len(params) != 3:
                raise QasmException(
                    'U called with {} params, instead of 3! '
                    'Params: {}, Args: {} at line {}'.format(
                        len(params), params, args, p.lineno(3)), self.qasm)
            if len(args) != 1:
                raise QasmException(
                    'U called with {} args, instead of 1! '
                    'Params: {}, Args: {} at line {}'.format(
                        len(args), params, args, p.lineno(5)), self.qasm)
            qreg = args[0]

            if len(qreg) > 1:
                raise QasmException(
                    'U called with quantum register instead of 1 qubit! '
                    'Params: {}, Args: {} at line {} '.format(
                        len(qreg), params, args, p.lineno(5)), self.qasm)

            p[0] = QasmUGate(params[1], params[0], params[2])(qreg[0])

    # params : parameter ',' params
    #        | parameter
    def p_params_multiple(self, p):
        """params : expr ',' params"""
        print('params multiple {} {} {} '.format(p[1], p[2], p[3]))
        p[3].insert(0, p[1])
        p[0] = p[3]
        print('   p[0] = {}'.format(p[0]))

    def p_params_single(self, p):
        """params : expr """
        print('params single {} '.format(p[1]))
        p[0] = [p[1]]

    # expr : term
    #            | func '(' expression ')' """
    #            | binary_op
    #            | unary_op
    def p_expr_term(self, p):
        """expr : term"""
        p[0] = p[1]

    def p_expr_binary(self, p):
        """expr : expr '*' expr
                | expr '/' expr
                | expr '+' expr
                | expr '-' expr
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

    # arg : ID '[' NATURAL_NUMBER ']'
    #

    def p_arg_bit(self, p):
        """arg : ID '[' NATURAL_NUMBER ']' """
        reg = p[1]
        num = p[3]
        arg_name = str(reg) + "_" + str(num)
        if reg in self.qregs.keys():
            if arg_name not in self.qubits.keys():
                self.qubits[arg_name] = NamedQubit(arg_name)
            p[0] = [self.qubits[arg_name]]
        else:
            raise QasmException(
                'undefined quantum/classical register "{}" '
                'at line no: {}'.format(reg, p.lineno(1)),
                self.qasm)

    def p_arg_register(self, p):
        """arg : ID """
        reg = p[1]
        if reg in self.qregs.keys():
            qubits = []
            for num in range(self.qregs[reg]):
                arg_name = str(reg) + "_" + str(num)
                if arg_name not in self.qubits.keys():
                    self.qubits[arg_name] = NamedQubit(arg_name)
                qubits.append(self.qubits[arg_name])
            p[0] = qubits
        else:
            raise QasmException(
                'undefined quantum/classical register "{}" '
                'at line no: {}'.format(reg, p.lineno(1)),
                self.qasm)

    def p_error(self, p):
        if p is None:
            raise QasmException('Unexpected end of file', self.qasm)

        raise QasmException(
            "Syntax error: '{}' at line {}".format(p.value, p.lineno),
            self.qasm)

    def p_empty(self, p):
        """empty :"""

    def parse(self) -> Qasm:
        if self.parsedQasm is None:
            self.parsedQasm = self.parser.parse(lexer=QasmLexer(self.qasm))
        return self.parsedQasm
