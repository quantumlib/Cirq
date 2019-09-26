# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterator, Callable, Union

import sympy

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.cell import CellMaker, ExplicitOperationsCell
from cirq.contrib.quirk.cells.parse import parse_formula


def generate_all_single_qubit_rotation_cell_makers() -> Iterator[CellMaker]:

    # Fixed single qubit rotations.
    yield gate("H", ops.H)
    yield gate("X", ops.X)
    yield gate("Y", ops.Y)
    yield gate("Z", ops.Z)
    yield gate("X^½", ops.X**(1 / 2))
    yield gate("X^⅓", ops.X**(1 / 3))
    yield gate("X^¼", ops.X**(1 / 4))
    yield gate("X^⅛", ops.X**(1 / 8))
    yield gate("X^⅟₁₆", ops.X**(1 / 16))
    yield gate("X^⅟₃₂", ops.X**(1 / 32))
    yield gate("X^-½", ops.X**(-1 / 2))
    yield gate("X^-⅓", ops.X**(-1 / 3))
    yield gate("X^-¼", ops.X**(-1 / 4))
    yield gate("X^-⅛", ops.X**(-1 / 8))
    yield gate("X^-⅟₁₆", ops.X**(-1 / 16))
    yield gate("X^-⅟₃₂", ops.X**(-1 / 32))
    yield gate("Y^½", ops.Y**(1 / 2))
    yield gate("Y^⅓", ops.Y**(1 / 3))
    yield gate("Y^¼", ops.Y**(1 / 4))
    yield gate("Y^⅛", ops.Y**(1 / 8))
    yield gate("Y^⅟₁₆", ops.Y**(1 / 16))
    yield gate("Y^⅟₃₂", ops.Y**(1 / 32))
    yield gate("Y^-½", ops.Y**(-1 / 2))
    yield gate("Y^-⅓", ops.Y**(-1 / 3))
    yield gate("Y^-¼", ops.Y**(-1 / 4))
    yield gate("Y^-⅛", ops.Y**(-1 / 8))
    yield gate("Y^-⅟₁₆", ops.Y**(-1 / 16))
    yield gate("Y^-⅟₃₂", ops.Y**(-1 / 32))
    yield gate("Z^½", ops.Z**(1 / 2))
    yield gate("Z^⅓", ops.Z**(1 / 3))
    yield gate("Z^¼", ops.Z**(1 / 4))
    yield gate("Z^⅛", ops.Z**(1 / 8))
    yield gate("Z^⅟₁₆", ops.Z**(1 / 16))
    yield gate("Z^⅟₃₂", ops.Z**(1 / 32))
    yield gate("Z^⅟₆₄", ops.Z**(1 / 64))
    yield gate("Z^⅟₁₂₈", ops.Z**(1 / 128))
    yield gate("Z^-½", ops.Z**(-1 / 2))
    yield gate("Z^-⅓", ops.Z**(-1 / 3))
    yield gate("Z^-¼", ops.Z**(-1 / 4))
    yield gate("Z^-⅛", ops.Z**(-1 / 8))
    yield gate("Z^-⅟₁₆", ops.Z**(-1 / 16))

    # Dynamic single qubit rotations.
    yield gate("X^t", ops.X**sympy.Symbol('t'))
    yield gate("Y^t", ops.Y**sympy.Symbol('t'))
    yield gate("Z^t", ops.Z**sympy.Symbol('t'))
    yield gate("X^-t", ops.X**-sympy.Symbol('t'))
    yield gate("Y^-t", ops.Y**-sympy.Symbol('t'))
    yield gate("Z^-t", ops.Z**-sympy.Symbol('t'))
    yield gate("e^iXt", ops.Rx(2 * sympy.pi * sympy.Symbol('t')))
    yield gate("e^iYt", ops.Ry(2 * sympy.pi * sympy.Symbol('t')))
    yield gate("e^iZt", ops.Rz(2 * sympy.pi * sympy.Symbol('t')))
    yield gate("e^-iXt", ops.Rx(-2 * sympy.pi * sympy.Symbol('t')))
    yield gate("e^-iYt", ops.Ry(-2 * sympy.pi * sympy.Symbol('t')))
    yield gate("e^-iZt", ops.Rz(-2 * sympy.pi * sympy.Symbol('t')))

    # Classically parameterized single qubit rotations.
    yield formula_gate("X^ft", "sin(pi*t)", lambda e: ops.X**e)
    yield formula_gate("Y^ft", "sin(pi*t)", lambda e: ops.Y**e)
    yield formula_gate("Z^ft", "sin(pi*t)", lambda e: ops.Z**e)
    yield formula_gate("Rxft", "pi*t*t", ops.Rx)
    yield formula_gate("Ryft", "pi*t*t", ops.Ry)
    yield formula_gate("Rzft", "pi*t*t", ops.Rz)


def gate(identifier: str, cirq_gate: 'cirq.Gate') -> CellMaker:
    return CellMaker(
        identifier,
        size=cirq_gate.num_qubits(),
        func=lambda args: ExplicitOperationsCell([cirq_gate.on(*args.qubits)]))


def formula_gate(identifier: str, default_formula: str,
                 gate_func: Callable[[Union[sympy.Symbol, float]], cirq.Gate]
                ) -> CellMaker:
    return CellMaker(identifier,
                     size=gate_func(0).num_qubits(),
                     func=lambda args: ExplicitOperationsCell([
                         gate_func(parse_formula(args.value, default_formula)).
                         on(*args.qubits)
                     ]))
