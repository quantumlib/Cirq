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
from cirq.contrib.quirk.cells.explicit_operations_cell import ExplicitOperationsCell
from cirq.contrib.quirk.cells.cell import (
    CellMaker,
)


def generate_all_single_qubit_rotation_cells() -> Iterator[CellMaker]:

    # Fixed single qubit rotations.
    yield from reg_gate("H", gate=ops.H)
    yield from reg_gate("X", gate=ops.X)
    yield from reg_gate("Y", gate=ops.Y)
    yield from reg_gate("Z", gate=ops.Z)
    yield from reg_gate("X^½", gate=ops.X**(1 / 2))
    yield from reg_gate("X^⅓", gate=ops.X**(1 / 3))
    yield from reg_gate("X^¼", gate=ops.X**(1 / 4))
    yield from reg_gate("X^⅛", gate=ops.X**(1 / 8))
    yield from reg_gate("X^⅟₁₆", gate=ops.X**(1 / 16))
    yield from reg_gate("X^⅟₃₂", gate=ops.X**(1 / 32))
    yield from reg_gate("X^-½", gate=ops.X**(-1 / 2))
    yield from reg_gate("X^-⅓", gate=ops.X**(-1 / 3))
    yield from reg_gate("X^-¼", gate=ops.X**(-1 / 4))
    yield from reg_gate("X^-⅛", gate=ops.X**(-1 / 8))
    yield from reg_gate("X^-⅟₁₆", gate=ops.X**(-1 / 16))
    yield from reg_gate("X^-⅟₃₂", gate=ops.X**(-1 / 32))
    yield from reg_gate("Y^½", gate=ops.Y**(1 / 2))
    yield from reg_gate("Y^⅓", gate=ops.Y**(1 / 3))
    yield from reg_gate("Y^¼", gate=ops.Y**(1 / 4))
    yield from reg_gate("Y^⅛", gate=ops.Y**(1 / 8))
    yield from reg_gate("Y^⅟₁₆", gate=ops.Y**(1 / 16))
    yield from reg_gate("Y^⅟₃₂", gate=ops.Y**(1 / 32))
    yield from reg_gate("Y^-½", gate=ops.Y**(-1 / 2))
    yield from reg_gate("Y^-⅓", gate=ops.Y**(-1 / 3))
    yield from reg_gate("Y^-¼", gate=ops.Y**(-1 / 4))
    yield from reg_gate("Y^-⅛", gate=ops.Y**(-1 / 8))
    yield from reg_gate("Y^-⅟₁₆", gate=ops.Y**(-1 / 16))
    yield from reg_gate("Y^-⅟₃₂", gate=ops.Y**(-1 / 32))
    yield from reg_gate("Z^½", gate=ops.Z**(1 / 2))
    yield from reg_gate("Z^⅓", gate=ops.Z**(1 / 3))
    yield from reg_gate("Z^¼", gate=ops.Z**(1 / 4))
    yield from reg_gate("Z^⅛", gate=ops.Z**(1 / 8))
    yield from reg_gate("Z^⅟₁₆", gate=ops.Z**(1 / 16))
    yield from reg_gate("Z^⅟₃₂", gate=ops.Z**(1 / 32))
    yield from reg_gate("Z^⅟₆₄", gate=ops.Z**(1 / 64))
    yield from reg_gate("Z^⅟₁₂₈", gate=ops.Z**(1 / 128))
    yield from reg_gate("Z^-½", gate=ops.Z**(-1 / 2))
    yield from reg_gate("Z^-⅓", gate=ops.Z**(-1 / 3))
    yield from reg_gate("Z^-¼", gate=ops.Z**(-1 / 4))
    yield from reg_gate("Z^-⅛", gate=ops.Z**(-1 / 8))
    yield from reg_gate("Z^-⅟₁₆", gate=ops.Z**(-1 / 16))

    # Dynamic single qubit rotations.
    yield from reg_gate("X^t", gate=ops.X**sympy.Symbol('t'))
    yield from reg_gate("Y^t", gate=ops.Y**sympy.Symbol('t'))
    yield from reg_gate("Z^t", gate=ops.Z**sympy.Symbol('t'))
    yield from reg_gate("X^-t", gate=ops.X**-sympy.Symbol('t'))
    yield from reg_gate("Y^-t", gate=ops.Y**-sympy.Symbol('t'))
    yield from reg_gate("Z^-t", gate=ops.Z**-sympy.Symbol('t'))
    yield from reg_gate("e^iXt", gate=ops.Rx(2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^iYt", gate=ops.Ry(2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^iZt", gate=ops.Rz(2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^-iXt",
                        gate=ops.Rx(-2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^-iYt",
                        gate=ops.Ry(-2 * sympy.pi * sympy.Symbol('t')))
    yield from reg_gate("e^-iZt",
                        gate=ops.Rz(-2 * sympy.pi * sympy.Symbol('t')))

    # Classically parameterized single qubit rotations.
    yield from reg_formula_gate("X^ft", "sin(pi*t)", lambda e: ops.X**e)
    yield from reg_formula_gate("Y^ft", "sin(pi*t)", lambda e: ops.Y**e)
    yield from reg_formula_gate("Z^ft", "sin(pi*t)", lambda e: ops.Z**e)
    yield from reg_formula_gate("Rxft", "pi*t*t", lambda e: ops.Rx(e))
    yield from reg_formula_gate("Ryft", "pi*t*t", lambda e: ops.Ry(e))
    yield from reg_formula_gate("Rzft", "pi*t*t", lambda e: ops.Rz(e))


def reg_gate(identifier: str, gate: 'cirq.Gate',
             basis_change: 'cirq.Gate' = None) -> Iterator[CellMaker]:
    yield CellMaker(
        identifier, gate.num_qubits(), lambda args: ExplicitOperationsCell(
            [gate.on(*args.qubits)],
            basis_change=[basis_change.on(*args.qubits)]
            if basis_change else ()))


def reg_formula_gate(
        identifier: str, default_formula: str,
        gate_func: Callable[[Union[sympy.Symbol, float]], cirq.Gate]
) -> Iterator[CellMaker]:
    from cirq.contrib.quirk.quirk_gate_reg_utils import parse_formula
    yield CellMaker(
        identifier,
        gate_func(0).num_qubits(), lambda args: ExplicitOperationsCell([
            gate_func(parse_formula(args.value, default_formula)).on(*args.
                                                                     qubits)
        ]))
