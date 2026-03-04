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

import dataclasses
import operator
from typing import Callable, TYPE_CHECKING
from enum import StrEnum

import numpy as np
import sympy
from ply import yacc

from cirq import Circuit, CircuitOperation, CX, FrozenCircuit, NamedQubit, ops, value, global_phase_operation
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import._lexer3 import Qasm3Lexer
from cirq.contrib.qasm_import.exception import QasmException
from cirq.contrib.qasm_import._parser import _generate_op_qubits

if TYPE_CHECKING:
    import cirq


class Qasm:
    """Qasm stores the final result of the Qasm parsing."""

    def __init__(
        self, supported_format: bool, std_gates_include: bool, regs: QasmRegisters, c: Circuit
    ):
        # defines whether the Quantum Experience standard header
        # is present or not
        self.std_gates_included = std_gates_include
        # defines if it has a supported format or not
        self.supportedFormat = supported_format
        # circuit
        self.regs = regs
        self.circuit = c

def _global_phase_operation(phase):
    if isinstance(phase, sympy.Expr):
        return ops.global_phase_operation(sympy.exp(1j * phase))
    return ops.global_phase_operation(np.exp(1j* phase))

class Qasm3UGate(QasmUGate):
    def _decompose_(self, qubits):
        gates: list = super()._decompose_(qubits)
        # add the additional phase introduced in QASM 3
        gates.append(_global_phase_operation(np.pi*self.theta/2))
        return gates

class QasmGateStatement:
    """Specifies how to convert a call to an OpenQASM gate
    to a list of `cirq.GateOperation`s.

    Has the responsibility to validate the arguments
    and parameters of the call and to generate a list of corresponding
    `cirq.GateOperation`s in the `on` method.
    """

    def __init__(
        self,
        qasm_gate: str,
        cirq_gate: ops.Gate | Callable[[list[float]], ops.Gate],
        num_params: int,
        num_args: int,
    ):
        """Initializes a Qasm gate statement.

        Args:
            qasm_gate: The symbol of the QASM gate.
            cirq_gate: The gate class on the cirq side.
            num_params: The number of params taken by this gate.
            num_args: The number of qubits (used in validation) this
                gate takes.
        """
        self.qasm_gate = qasm_gate
        self.cirq_gate = cirq_gate
        self.num_params = num_params

        # at least one quantum argument is mandatory for gates to act on
        assert num_args >= 1
        self.num_args = num_args

    def _validate_params(self, params: list[value.TParamVal], lineno: int):
        if len(params) != self.num_params:
            raise QasmException(
                f"{self.qasm_gate} takes {self.num_params} parameter(s), "
                f"got: {len(params)}, at line {lineno}"
            )
        
    def resolve(self, params: list[value.TParamVal], lineno:int):
        self._validate_params(params, lineno)
        # the actual gate we'll apply the arguments to might be a parameterized
        # or non-parameterized gate
        final_gate: ops.Gate = (
            self.cirq_gate if isinstance(self.cirq_gate, ops.Gate) else self.cirq_gate(params)
        )
        return final_gate


class CustomGate:
    """Represents an invocation of a user-defined gate.

    The custom gate definition is encoded here as a `FrozenCircuit`, and the
    arguments (params and qubits) of the specific invocation of that gate are
    stored here too. When `on` is called, we create a CircuitOperation, mapping
    the qubits and params to the values provided."""

    name: str
    circuit: FrozenCircuit
    params: tuple[str, ...]
    qubits: tuple[ops.Qid, ...]

    class CustomQasmGate(ops.Gate):
        def __init__(self, name:str, sub_circuit: ops.CircuitOperation):
            self.sub_circuit = sub_circuit
            self.name = name

        def _num_qubits_(self):
            return self.sub_circuit._num_qubits_()

        def _decompose_(self, qubits):
            return self.sub_circuit.with_qubits(*qubits)

        def __repr__(self):
            return f"<CustomQasmGate.{self.name}>"

    def __init__(self, name, circuit, params, qubits):
        self.name = name
        self.circuit = circuit
        self.params = params
        self.qubits = qubits
        super(CustomGate, self)

    def resolve(
        self, params: list[value.TParamVal], lineno: int
    ) -> ops.Gate:
        if len(params) != len(self.params):
            raise QasmException(f"Wrong number of params for '{self.name}' at line {lineno}")
        rescircuit = CircuitOperation(
            self.circuit,
            param_resolver=dict(zip(self.params, params)),
        )
        return CustomGate.CustomQasmGate(self.name, rescircuit)


class RegisterType(StrEnum):
    QUBIT = 'qubit'
    BIT = 'bit'
    ANGLE = 'angle'


@dataclasses.dataclass
class Register:
    name: str
    type_: RegisterType
    length: int


@dataclasses.dataclass
class QasmRegisters:
    qubits: dict[str, int] = dataclasses.field(default_factory=dict)
    bits: dict[str, int] = dataclasses.field(default_factory=dict)
    angles: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def defined_regs(self) -> list[str]:
        return self.classical_regs + self.quantum_regs
        
    @property
    def classical_regs(self) -> list[str]:
        return list(self.bits.keys()) +\
        list(self.angles.keys())
    
    @property
    def quantum_regs(self) -> list[str]:
        return list(self.qubits.keys())
    
    def add_register(self, register: Register, lineno):
        if register.name in self.defined_regs:
            raise QasmException(f"{register.name} is already defined at line {lineno}")
        if register.length == 0:
            raise QasmException(f"Illegal, zero-length register '{register.name}' at line {lineno}")
        match register.type_:
            case RegisterType.QUBIT:
                registers = self.qubits
            case RegisterType.BIT:
                registers = self.bits
            case RegisterType.ANGLE:
                registers = self.angles
        registers[register.name] = register.length

    def __getitem__(self, name:str)-> Register | None:
        if name in self.qubits:
            type_ = RegisterType.QUBIT
            registers = self.qubits
        elif name in self.bits:
            type_ = RegisterType.BIT
            registers = self.bits
        elif name in self.angles:
            type_ = RegisterType.ANGLE
            registers = self.angles
        else:
            return
        return Register(name, type_, registers[name])


class Qasm3Parser:
    """Parser for QASM3 strings.

    Example:

        qasm = "OPENQASM 2.0; qreg q1[2]; CX q1[0], q1[1];"
        parsedQasm = QasmParser().parse(qasm)
    """

    def __init__(self) -> None:
        self.circuit_stack: list[Circuit] = [Circuit()]
        self.register_stack: list[QasmRegisters] = [QasmRegisters()]
        self.in_global_scope: bool = True
        self.in_gate_or_function_scope: bool = False
        self.gate_set: dict[str, CustomGate | QasmGateStatement] = {**self.basic_gates}
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
        self.lexer = Qasm3Lexer()
        self.stdgatesinc = False
        self.supported_format = False
        self.parsedQasm: Qasm | None = None
        self.qubits: dict[str, ops.Qid] = {} # TODO could move into regs

        self.binary_operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow,
        }
        self.functions = {
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'arccos': np.acos,
            'arctan': np.atan,
            'arcsin': np.asin,
            'ceiling': np.ceil,
            'floor': np.floor,
            'mod': np.mod,
        }

    basic_gates: dict[str, QasmGateStatement] = {
        'U': QasmGateStatement(
            qasm_gate='U',
            num_params=3,
            num_args=1,
            # QasmUGate expects half turns
            cirq_gate=(lambda params: Qasm3UGate(*[p / np.pi for p in params])),
        ),
    }

    std_gates = {
        'p': QasmGateStatement(
            qasm_gate='p',
            cirq_gate=(lambda params: ops.ZPowGate(exponent=params[0] / np.pi)),
            num_params=1,
            num_args=1,
        ),
        'x': QasmGateStatement(qasm_gate='x', num_params=0, num_args=1, cirq_gate=ops.X),
        'y': QasmGateStatement(qasm_gate='y', num_params=0, num_args=1, cirq_gate=ops.Y),
        'z': QasmGateStatement(qasm_gate='z', num_params=0, num_args=1, cirq_gate=ops.Z),
        'h': QasmGateStatement(qasm_gate='h', num_params=0, num_args=1, cirq_gate=ops.H),
        's': QasmGateStatement(qasm_gate='s', num_params=0, num_args=1, cirq_gate=ops.S),
        'sdg': QasmGateStatement(qasm_gate='sdg', num_params=0, num_args=1, cirq_gate=ops.S**-1),
        't': QasmGateStatement(qasm_gate='t', num_params=0, num_args=1, cirq_gate=ops.T),
        'tdg': QasmGateStatement(qasm_gate='tdg', num_params=0, num_args=1, cirq_gate=ops.T**-1),
        'sx': QasmGateStatement(
            qasm_gate='sx', num_params=0, num_args=1, cirq_gate=ops.XPowGate(exponent=0.5)
        ),
        'rx': QasmGateStatement(
            qasm_gate='rx', cirq_gate=(lambda params: ops.rx(params[0])), num_params=1, num_args=1
        ),
        'ry': QasmGateStatement(
            qasm_gate='ry', cirq_gate=(lambda params: ops.ry(params[0])), num_params=1, num_args=1
        ),
        'rz': QasmGateStatement(
            qasm_gate='rz', cirq_gate=(lambda params: ops.rz(params[0])), num_params=1, num_args=1
        ),
        'cx': QasmGateStatement(qasm_gate='cx', cirq_gate=CX, num_params=0, num_args=2),
        'cy': QasmGateStatement(qasm_gate='cy', cirq_gate=ops.CY, num_params=0, num_args=2),
        'cz': QasmGateStatement(qasm_gate='cz', cirq_gate=ops.CZ, num_params=0, num_args=2),
        'cp': QasmGateStatement(
            qasm_gate='cp',
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: ops.ControlledGate(ops.ZPowGate(exponent=params[0] / np.pi))),
        ),
        'crx': QasmGateStatement(
            qasm_gate='crx',
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: ops.ControlledGate(ops.rx(params[0]))),
        ),
        'cry': QasmGateStatement(
            qasm_gate='cry',
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: ops.ControlledGate(ops.ry(params[0]))),
        ),
        'crz': QasmGateStatement(
            qasm_gate='crz',
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: ops.ControlledGate(ops.rz(params[0]))),
        ),
        'ch': QasmGateStatement(
            qasm_gate='ch', cirq_gate=ops.ControlledGate(ops.H), num_params=0, num_args=2
        ),
        'swap': QasmGateStatement(qasm_gate='swap', cirq_gate=ops.SWAP, num_params=0, num_args=2),
        'ccx': QasmGateStatement(qasm_gate='ccx', num_params=0, num_args=3, cirq_gate=ops.CCX),
        'cswap': QasmGateStatement(
            qasm_gate='cswap', num_params=0, num_args=3, cirq_gate=ops.CSWAP
        ),
        'cu': QasmGateStatement(
            qasm_gate='cu',
            num_params=4,
            num_args=2,
            cirq_gate=(
                lambda params: ops.ControlledGate(
                    ops.MatrixGate((QasmUGate(params[0]/np.pi, params[1]/np.pi, params[2]/np.pi)*np.exp(1j*params[3])).matrix())
                )
            ),
        ),
        'CX': QasmGateStatement(
            qasm_gate='CX',
            num_params=0,
            num_args=2,
            cirq_gate=ops.ControlledGate(Qasm3UGate(*[np.pi, 0, np.pi])),
        ),
        'phase': QasmGateStatement(
            qasm_gate='phase',
            num_params=1,
            num_args=1,
            cirq_gate=(lambda params: Qasm3UGate(*[0, 0, params[0]])),
        ),
        'cphase': QasmGateStatement(
            qasm_gate='cphase',
            num_params=1,
            num_args=2,
            cirq_gate=(lambda params: ops.ControlledGate(Qasm3UGate(*[0, 0, params[0]]))),
        ),
        'id': QasmGateStatement(
            qasm_gate='id', cirq_gate=ops.IdentityGate(1), num_params=0, num_args=1
        ),
        'u1': QasmGateStatement(
            qasm_gate='u1',
            cirq_gate=(lambda params: QasmUGate(0, 0, params[0] / np.pi)),
            num_params=1,
            num_args=1,
        ),
        'u2': QasmGateStatement(
            qasm_gate='u2',
            cirq_gate=(lambda params: QasmUGate(0.5, params[0] / np.pi, params[1] / np.pi)),
            num_params=2,
            num_args=1,
        ),
        'u3': QasmGateStatement(
            qasm_gate='u3',
            num_params=3,
            num_args=1,
            cirq_gate=(lambda params: QasmUGate(*[p / np.pi for p in params])),
        ),
    }

    tokens = Qasm3Lexer.tokens
    start = 'start'

    precedence = (
        ('left', '+', '-'), 
        ('left', '*', '/'), 
        ('right', 'DOUBLE_ASTERISK'),
    )

    @property
    def regs(self) -> QasmRegisters:
        return self.register_stack[-1]
    
    @property
    def circuit(self) -> Circuit:
        return self.circuit_stack[-1]

    def p_start(self, p):
        """start : version qasm
        | version
        """
        p[0] = Qasm(self.supported_format, self.stdgatesinc, self.regs, self.circuit)

    def p_version(self, p):
        """version : FORMAT_SPEC"""
        if p[1] not in ["3.0"]:
            raise QasmException(
                f"Unsupported OpenQASM version: {p[1]}, "
                "only 3.0 is supported currently by Cirq"
            )
        self.supported_format = True

    def p_no_format_specified_error(self, p):
        """version : qasm"""
        if self.supported_format is False:
            raise QasmException("Missing 'OPENQASM 3.0;' statement")
    
    def p_scope_full(self, p):
        """scope : '{' qasm '}'
        """
        p[0] = p[2]

    def p_scope_empty(self, p):
        """scope : '{' '}'
        """

    def p_qasm(self, p):
        """qasm : statementOrScope qasm
        | statementOrScope
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[2]

    def p_statementOrScope_statement(self, p):
        """statementOrScope : statement
        """
        p[0] = p[1]

    def p_statement_qasm(self, p):
        """statement : classicalDeclarationStatement
        | gateStatement
        | includeStatement
        | oldStyleDeclarationStatement
        | quantumDeclarationStatement
        | empty
        """
        p[0] = p[1]

    def p_statement_circuit(self, p):
        """statement : gateCallStatement
        | measureArrowAssignmentStatement
        | measureAssignmentStatement
        | resetStatement 
        """
        self.circuit.append(p[1])
        p[0] = p[1]

    def p_includeStatement_stdgates(self, p):
        """includeStatement : STDGATESINC
        """
        self.stdgatesinc = True
        self.gate_set.update(self.std_gates)

    def p_resetStatement(self, p):
        """resetStatement : RESET gateOperand SEMI"""
        qreg = p[2]
        p[0] = [ops.ResetChannel().on(qreg[i]) for i in range(len(qreg))]

    def p_gateCall_base(self, p):
        """gateCall : ID '(' expressionList ')'
        | ID
        """
        gate_name = p[1]
        params = [] if len(p) != 5 else p[3]
        if gate_name not in self.gate_set:
            tip = ", did you forget to include stdgates.inc?" if not self.stdgatesinc else ""
            msg = f'Unknown gate "{gate_name}" at line {p.lineno(1)}{tip}'
            raise QasmException(msg)
        gate = self.gate_set[gate_name].resolve(params=params, lineno=p.lineno(1))
        p[0] = (gate_name, gate)

    def p_gateCall_modified(self, p):
        """gateCall : gateModifier gateCall
        """
        p[0] = (p[2][0], p[1](p[2][1]))

    def p_gateCallStatement(self, p):
        """gateCallStatement : gateCall gateOperandList SEMI
        """
        args = p[2]
        gate = p[1][1]
        gate_name = p[1][0]
        if len(args) != gate.num_qubits():
            raise QasmException(
                f"{gate_name} only takes {gate.num_qubits()} arg(s) (qubits and/or registers), "
                f"got: {len(args)}, at line {p.lineno(3)}"
            )
        p[0] = (gate.on(*qubits) for qubits in _generate_op_qubits(args, p.lineno(3)))
    
    def p_gateCallStatement_gphase(self, p):
        """gateCallStatement : GPHASE '(' expressionList ')' SEMI
        | GPHASE expressionList SEMI
        """
        args = p[2] if len(p) == 4 else p[3]
        if len(args) != 1:
            raise QasmException(
                f"gphase takes 1 parameter, "
                f"got: {len(args)}, at line {p.lineno(1)}"
            )
        phase = args[0]
        p[0] = _global_phase_operation(phase)

    def p_gateCall_ctrl_gphase(self, p):
        """gateCall : CTRL AT GPHASE '(' expressionList ')'
        """
        args = p[5]
        if len(args) != 1:
            raise QasmException(
                f"gphase takes 1 parameter, "
                f"got: {len(args)}, at line {p.lineno(1)}"
            )
        phase = args[0]
        p[0] = ('gphase', ops.ZPowGate(exponent=phase / np.pi))

    def p_measureArrowAssignmentStatement(self, p):
        """measureArrowAssignmentStatement : measureExpression ARROW indexedIdentifier SEMI
        | measureExpression SEMI 
        """
        p[0] = self._measure(p[3], p[1], p.lineno(2))
    
    def p_classicalDeclarationStatement_scalar(self, p):
        """classicalDeclarationStatement : scalarType ID SEMI
        """
        reg = Register(p[2], *p[1])
        self.regs.add_register(reg,  p.lineno(2))
        p[0] = reg

    def p_oldStyleDeclarationStatement(self, p):
        """oldStyleDeclarationStatement : CREG ID designator SEMI
        | CREG ID SEMI
        | QREG ID designator SEMI
        | QREG ID SEMI
        """
        name = p[2]
        length = 1 if len(p) == 4 else p[3]
        if p[1] == 'qreg':
            if not self.in_global_scope:
                raise QasmException(
                    f"Qubits cannot be declared in non global scope at line {p.lineno(1)}"
                )
            type_ = RegisterType.QUBIT
        else:
            type_ = RegisterType.BIT
        reg = Register(name, type_, length)
        self.regs.add_register(reg,  p.lineno(2))
        p[0] = reg

    def p_quantumDeclarationStatement(self, p):
        """quantumDeclarationStatement : qubitType ID SEMI"""
        if not self.in_global_scope:
            raise QasmException(
                f"Qubits cannot be declared in non global scope at line {p.lineno(2)}"
            )
        type_, length = p[1]
        name = p[2]
        reg = Register(name, type_, length)
        self.regs.add_register(reg,  p.lineno(2))
        p[0] = reg

    def p_gateParams_identifierList(self, p):
        """gateParams_identifierList : identifierList"""
        for reg in p[1]:
            self.regs.add_register(Register(reg, RegisterType.ANGLE, 1),  p.lineno(1))
        p[0] = p[1]

    def p_gateQubits_identifierList(self, p):
        """gateQubits_identifierList : identifierList"""
        for reg in p[1]:
            self.regs.add_register(Register(reg, RegisterType.QUBIT, 1), p.lineno(1))
        p[0] = p[1]

    def _gate_def(self, name: str, gate_params: list, gate_qubits: list, lineno: int):
        circuit = Circuit(self.circuit).freeze()
        gate_def = CustomGate(name, circuit, gate_params, gate_qubits)
        self.gate_set[name] = gate_def
        self.register_stack.pop()
        self.circuit_stack.pop()
        self.in_global_scope = True
        self.in_gate_or_function_scope = False
        return gate_def
    
    def p_enter_custom_gate_scope(self, p):
        """enter_custom_gate_scope : GATE"""
        if not self.in_global_scope:
            raise QasmException(f"Custom gates cannot be defined in non global scope at line {p.lineno(1)}")
        self.in_global_scope = False
        self.in_gate_or_function_scope = True
        self.register_stack.append(QasmRegisters())
        self.circuit_stack.append(Circuit())

    def p_gateStatement_params(self, p):
        """gateStatement : enter_custom_gate_scope ID '(' gateParams_identifierList ')' gateQubits_identifierList scope
        """
        self._gate_def(p[2], p[4], p[6], p.lineno(1))

    def p_gateStatement_no_params(self, p):
        """gateStatement : enter_custom_gate_scope ID '(' ')' gateQubits_identifierList scope
        | enter_custom_gate_scope ID gateQubits_identifierList scope
        """
        offset = len(p) - 6
        self._gate_def(p[2], [], p[3+offset], p.lineno(1))   

    def p_assignmentStatement_measure(self, p):
        """measureAssignmentStatement : indexedIdentifier '=' measureExpression SEMI
        """
        p[0] = self._measure(p[1], p[3], p.lineno(2))

    def _measure(self, cregs: tuple[Register, list[int]], qregs:list[ops.Qid], lineno):
        creg_type = cregs[0].type_
        if  creg_type != RegisterType.BIT:
            raise QasmException(
                f"Illegal use of `{creg_type}` type register for measurement results at line {lineno}"
            )
        if len(qregs) != len(cregs[1]):
            raise QasmException(
                f'mismatched register sizes {len(qregs)} -> {len(cregs[1])} for measurement '
                f'at line {lineno}'
            )
        creg = [self.make_name(i, cregs[0].name) for i in cregs[1]]
        return [
            ops.MeasurementGate(num_qubits=1, key=creg[i]).on(qregs[i]) for i in range(len(qregs))
        ]

    # Second level defs

    def p_expression_binary_operator(self, p):
        """expression : expression '+' expression
        | expression '-' expression
        | expression '*' expression
        | expression '/' expression
        | expression DOUBLE_ASTERISK expression
        """
        p[0] = self.binary_operators[p[2]](p[1], p[3])

    def p_expression_call(self, p):
        """expression : ID '(' expressionList ')'
        """
        func = p[1]
        args = p[3]
        if func not in self.functions:
            raise QasmException(
                f"Function not recognized: '{func}' at line {p.lineno(1)}"
            )
        if any( not isinstance(exp, (int, float)) for exp in args):
            raise QasmException(
                f"Non contant input for built-in function `{func}` at line {p.lineno(1)}"
            )
        p[0] = self.functions[func](*args)

    def p_parenExp(self, p):
        """expression : '(' expression ')'"""
        p[0] = p[2]

    def p_unaryExp(self, p):
        """expression : '-' expression
        """
        p[0] = -p[2]

    def p_literalExp_number(self, p):
        """expression : NUMBER
        | NATURAL_NUMBER
        """
        p[0] = p[1]

    def p_literalExp_id(self, p):
        """expression : ID
        """
        if p[1] not in self.regs.defined_regs:
            raise QasmException(f"Undefined parameter '{p[1]}' in line {p.lineno(1)}")
        p[0] = sympy.Symbol(p[1])
    
    def p_pi(self, p):
        """expression : PI"""
        p[0] = np.pi
    
    def p_measureExpression(self, p):
        """measureExpression : MEASURE gateOperand"""
        p[0] = p[2]

    def p_indexOperator(self, p):
        """indexOperator : '[' NATURAL_NUMBER ']'"""
        p[0] = [p[2]]

    def p_indexedIdentifier(self, p):
        """indexedIdentifier : ID indexOperator
        | ID
        """
        reg = p[1]
        if reg not in self.regs.defined_regs:
            raise QasmException(f"Undefined register '{reg}' at line {p.lineno(1)}")
        reg =  self.regs[reg]
        size = reg.length
        idxs = list(range(size))
        if len(p) == 3:
            if reg.type_ == RegisterType.QUBIT and self.in_gate_or_function_scope:
                raise QasmException(
                    f"Cannot index into qubit while defining a custom gate at line {p.lineno(1)}"
                )
            for idx in p[2]:
                if idx >= size:
                    raise QasmException(
                        f'Out of bounds index {idx} '
                        f'on register {reg.name} of size {size} '
                        f'at line {p.lineno(1)}'
                    )
            idxs = p[2]
        p[0] = (reg, idxs)

#####

    def p_gateModifier_ctrl(self, p):
        """gateModifier : CTRL AT
        | CTRL '(' expression ')' AT
        """
        n_controls = 1 
        if len(p) == 6:
            if not isinstance(p[3], int):
                raise QasmException(f"Number of control qubits must be an integer known at compile time at line {p.lineno(1)}")
            n_controls = p[3]
        def ctrl(gate:ops.Gate) -> ops.ControlledGate:
            return ops.ControlledGate(gate, n_controls)
        p[0] = ctrl

    def p_scalarType_bit(self, p):
        """scalarType : BIT designator
        | BIT
        """
        length = 1 if len(p) != 3 else p[2]
        p[0] = (RegisterType.BIT, length)

    def p_scalarType_angle(self, p):
        """scalarType : ANGLE designator
        | ANGLE
        """
        length = 1 if len(p) != 3 else p[2]
        p[0] = (RegisterType.ANGLE, length)

    def p_qubitType(self, p):
        """qubitType : QUBIT
        | QUBIT designator
        """
        length = 1 if len(p) != 3 else p[2]
        p[0] = (RegisterType.QUBIT, length)

    def p_designator(self, p):
        """designator : '[' NATURAL_NUMBER ']'"""
        p[0] = p[2]

    def p_gateOperand(self, p):
        """gateOperand : indexedIdentifier"""
        reg: Register = p[1][0]
        if  reg.type_ != RegisterType.QUBIT:
            raise QasmException(
                f"Illegal use of {reg.type_} register where qubit required at line {p.lineno(1)}"
            )
        qubits = []
        for idx in p[1][1]:
            arg_name = self.make_name(idx, reg.name)
            if arg_name not in self.qubits.keys():
                self.qubits[arg_name] = NamedQubit(arg_name)
            qubits.append(self.qubits[arg_name])
        p[0] = qubits

    def make_name(self, idx, reg):
        return str(reg) + "_" + str(idx)

    def p_expressionList(self, p):
        """expressionList : expression ',' expressionList
        | expression
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[3] 

    def p_identifierList(self, p):
        """identifierList : ID ',' identifierList
        | ID
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[3] 

    def p_gateOperandList(self, p):
        """gateOperandList : gateOperand ',' gateOperandList
        | gateOperand
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = [p[1]] + p[3] 
    
#####################

    def p_error(self, p):
        if p is None:
            raise QasmException('Unexpected end of file')

        raise QasmException(f"""Syntax error: '{p.value}'
{self.debug_context(p)}
at line {p.lineno}, column {self.find_column(p)}""")

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
        debug_end = min(self.qasm.find('\n', p.lexpos, p.lexpos + 5), p.lexpos + 5)

        return (
            "..."
            + self.qasm[debug_start:debug_end]
            + "\n"
            + (" " * (3 + p.lexpos - debug_start))
            + "^"
        )
