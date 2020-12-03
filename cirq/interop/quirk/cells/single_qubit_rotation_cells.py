# Copyright 2019 The Cirq Developers
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
from typing import Iterator, Callable, Union, TYPE_CHECKING

import sympy

from cirq import ops
from cirq.interop.quirk.cells.cell import CellMaker
from cirq.interop.quirk.cells.parse import parse_formula

if TYPE_CHECKING:
    import cirq


def generate_all_single_qubit_rotation_cell_makers() -> Iterator[CellMaker]:

    # Fixed single qubit rotations.
    yield _gate("H", ops.H)
    yield _gate("X", ops.X)
    yield _gate("Y", ops.Y)
    yield _gate("Z", ops.Z)
    yield _gate("X^½", ops.X ** (1 / 2))
    yield _gate("X^⅓", ops.X ** (1 / 3))
    yield _gate("X^¼", ops.X ** (1 / 4))
    yield _gate("X^⅛", ops.X ** (1 / 8))
    yield _gate("X^⅟₁₆", ops.X ** (1 / 16))
    yield _gate("X^⅟₃₂", ops.X ** (1 / 32))
    yield _gate("X^-½", ops.X ** (-1 / 2))
    yield _gate("X^-⅓", ops.X ** (-1 / 3))
    yield _gate("X^-¼", ops.X ** (-1 / 4))
    yield _gate("X^-⅛", ops.X ** (-1 / 8))
    yield _gate("X^-⅟₁₆", ops.X ** (-1 / 16))
    yield _gate("X^-⅟₃₂", ops.X ** (-1 / 32))
    yield _gate("Y^½", ops.Y ** (1 / 2))
    yield _gate("Y^⅓", ops.Y ** (1 / 3))
    yield _gate("Y^¼", ops.Y ** (1 / 4))
    yield _gate("Y^⅛", ops.Y ** (1 / 8))
    yield _gate("Y^⅟₁₆", ops.Y ** (1 / 16))
    yield _gate("Y^⅟₃₂", ops.Y ** (1 / 32))
    yield _gate("Y^-½", ops.Y ** (-1 / 2))
    yield _gate("Y^-⅓", ops.Y ** (-1 / 3))
    yield _gate("Y^-¼", ops.Y ** (-1 / 4))
    yield _gate("Y^-⅛", ops.Y ** (-1 / 8))
    yield _gate("Y^-⅟₁₆", ops.Y ** (-1 / 16))
    yield _gate("Y^-⅟₃₂", ops.Y ** (-1 / 32))
    yield _gate("Z^½", ops.Z ** (1 / 2))
    yield _gate("Z^⅓", ops.Z ** (1 / 3))
    yield _gate("Z^¼", ops.Z ** (1 / 4))
    yield _gate("Z^⅛", ops.Z ** (1 / 8))
    yield _gate("Z^⅟₁₆", ops.Z ** (1 / 16))
    yield _gate("Z^⅟₃₂", ops.Z ** (1 / 32))
    yield _gate("Z^⅟₆₄", ops.Z ** (1 / 64))
    yield _gate("Z^⅟₁₂₈", ops.Z ** (1 / 128))
    yield _gate("Z^-½", ops.Z ** (-1 / 2))
    yield _gate("Z^-⅓", ops.Z ** (-1 / 3))
    yield _gate("Z^-¼", ops.Z ** (-1 / 4))
    yield _gate("Z^-⅛", ops.Z ** (-1 / 8))
    yield _gate("Z^-⅟₁₆", ops.Z ** (-1 / 16))

    # Dynamic single qubit rotations.
    yield _gate("X^t", ops.X ** sympy.Symbol('t'))
    yield _gate("Y^t", ops.Y ** sympy.Symbol('t'))
    yield _gate("Z^t", ops.Z ** sympy.Symbol('t'))
    yield _gate("X^-t", ops.X ** -sympy.Symbol('t'))
    yield _gate("Y^-t", ops.Y ** -sympy.Symbol('t'))
    yield _gate("Z^-t", ops.Z ** -sympy.Symbol('t'))
    yield _gate("e^iXt", ops.rx(2 * sympy.pi * sympy.Symbol('t')))
    yield _gate("e^iYt", ops.ry(2 * sympy.pi * sympy.Symbol('t')))
    yield _gate("e^iZt", ops.rz(2 * sympy.pi * sympy.Symbol('t')))
    yield _gate("e^-iXt", ops.rx(-2 * sympy.pi * sympy.Symbol('t')))
    yield _gate("e^-iYt", ops.ry(-2 * sympy.pi * sympy.Symbol('t')))
    yield _gate("e^-iZt", ops.rz(-2 * sympy.pi * sympy.Symbol('t')))

    # Formulaic single qubit rotations.
    yield _formula_gate("X^ft", "sin(pi*t)", lambda e: ops.X ** e)
    yield _formula_gate("Y^ft", "sin(pi*t)", lambda e: ops.Y ** e)
    yield _formula_gate("Z^ft", "sin(pi*t)", lambda e: ops.Z ** e)
    yield _formula_gate("Rxft", "pi*t*t", ops.rx)
    yield _formula_gate("Ryft", "pi*t*t", ops.ry)
    yield _formula_gate("Rzft", "pi*t*t", ops.rz)


def _gate(identifier: str, gate: 'cirq.Gate') -> CellMaker:
    return CellMaker(
        identifier=identifier, size=gate.num_qubits(), maker=lambda args: gate.on(*args.qubits)
    )


def _formula_gate(
    identifier: str,
    default_formula: str,
    gate_func: Callable[[Union[sympy.Symbol, float]], 'cirq.Gate'],
) -> CellMaker:
    return CellMaker(
        identifier=identifier,
        size=gate_func(0).num_qubits(),
        maker=lambda args: gate_func(
            parse_formula(default_formula if args.value is None else args.value)
        ).on(*args.qubits),
    )
