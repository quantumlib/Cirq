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
from typing import Iterator

import sympy

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.arithmetic_cells import all_arithmetic_cells
from cirq.contrib.quirk.cells.cell import (
    CellMaker,
)
from cirq.contrib.quirk.cells.control_cells import all_control_cells
from cirq.contrib.quirk.cells.input_cells import (
    all_input_cells
)
from cirq.contrib.quirk.cells.qubit_permutation_cells import \
    all_qubit_permutation_cells
from cirq.contrib.quirk.cells.single_qubit_rotation_cells import \
    generate_all_single_qubit_rotation_cells
from cirq.contrib.quirk.cells.swap_cell import (
    SwapCell,
)


def generate_all_cells() -> Iterator[CellMaker]:
    from cirq.contrib.quirk.quirk_gate_reg_utils import (
        reg_unsupported_gates, reg_gate, reg_const, reg_formula_gate,
        reg_parameterized_gate, reg_ignored_family, reg_ignored_gate,
        reg_unsupported_family,
        reg_family,
        CellMaker,
        reg_measurement)

    # Swap.
    yield CellMaker("Swap",
                   1, lambda args: SwapCell(args.qubits, []))

    yield from all_control_cells()
    yield from all_input_cells()

    # Post selection.
    yield from reg_unsupported_gates(
        "|0⟩⟨0|",
        "|1⟩⟨1|",
        "|+⟩⟨+|",
        "|-⟩⟨-|",
        "|X⟩⟨X|",
        "|/⟩⟨/|",
        "0",
        reason='postselection is not implemented in Cirq')

    # Non-physical operations.
    yield from reg_unsupported_gates("__error__",
                                     "__unstable__UniversalNot",
                                     reason="Unphysical operation.")

    # Scalars.
    yield from reg_gate("…", gate=ops.IdentityGate(num_qubits=1))
    yield from reg_const("NeGate", ops.GlobalPhaseOperation(-1))
    yield from reg_const("i", ops.GlobalPhaseOperation(1j))
    yield from reg_const("-i", ops.GlobalPhaseOperation(-1j))
    yield from reg_const("√i", ops.GlobalPhaseOperation(1j**0.5))
    yield from reg_const("√-i", ops.GlobalPhaseOperation((-1j)**0.5))

    # Measurement.
    yield from reg_measurement("Measure")
    yield from reg_measurement("ZDetector")
    yield from reg_measurement("YDetector", basis_change=ops.X**-0.5)
    yield from reg_measurement("XDetector", basis_change=ops.Y**0.5)
    yield from reg_unsupported_gates(
        "XDetectControlReset",
        "YDetectControlReset",
        "ZDetectControlReset",
        reason="Classical feedback is not implemented in Cirq")

    yield from generate_all_single_qubit_rotation_cells()

    # Classically parameterized single qubit rotations.
    yield from reg_formula_gate("X^ft", "sin(pi*t)", lambda e: ops.X**e)
    yield from reg_formula_gate("Y^ft", "sin(pi*t)", lambda e: ops.Y**e)
    yield from reg_formula_gate("Z^ft", "sin(pi*t)", lambda e: ops.Z**e)
    yield from reg_formula_gate("Rxft", "pi*t*t", lambda e: ops.Rx(e))
    yield from reg_formula_gate("Ryft", "pi*t*t", lambda e: ops.Ry(e))
    yield from reg_formula_gate("Rzft", "pi*t*t", lambda e: ops.Rz(e))

    # Quantum parameterized single qubit rotations.
    yield from reg_parameterized_gate("X^(A/2^n)", ops.X, +1)
    yield from reg_parameterized_gate("Y^(A/2^n)", ops.Y, +1)
    yield from reg_parameterized_gate("Z^(A/2^n)", ops.Z, +1)
    yield from reg_parameterized_gate("X^(-A/2^n)", ops.X, -1)
    yield from reg_parameterized_gate("Y^(-A/2^n)", ops.Y, -1)
    yield from reg_parameterized_gate("Z^(-A/2^n)", ops.Z, -1)

    # Displays.
    yield from reg_ignored_family("Amps")
    yield from reg_ignored_family("Chance")
    yield from reg_ignored_family("Sample")
    yield from reg_ignored_family("Density")
    yield from reg_ignored_gate("Bloch")

    yield from all_arithmetic_cells()

    # Dynamic gates with discretized actions.
    yield from reg_unsupported_gates("X^⌈t⌉",
                                     "X^⌈t-¼⌉",
                                     reason="discrete parameter")
    yield from reg_unsupported_family("Counting", reason="discrete parameter")
    yield from reg_unsupported_family("Uncounting", reason="discrete parameter")
    yield from reg_unsupported_family(">>t", reason="discrete parameter")
    yield from reg_unsupported_family("<<t", reason="discrete parameter")

    # Gates that are no longer in the toolbox and have dominant replacements.
    yield from reg_unsupported_family("add",
                                      reason="deprecated; use +=A instead")
    yield from reg_unsupported_family("sub",
                                      reason="deprecated; use -=A instead")
    yield from reg_unsupported_family("c+=ab",
                                      reason="deprecated; use +=AB instead")
    yield from reg_unsupported_family("c-=ab",
                                      reason="deprecated; use -=AB instead")

    # Frequency space.
    yield from reg_family("QFT", lambda n: cirq.QuantumFourierTransformGate(n))
    yield from reg_family(
        "QFT†", lambda n: cirq.inverse(cirq.QuantumFourierTransformGate(n)))
    yield from reg_family(
        "PhaseGradient", lambda n: cirq.PhaseGradientGate(num_qubits=n,
                                                          exponent=0.5))
    yield from reg_family(
        "PhaseUngradient", lambda n: cirq.PhaseGradientGate(num_qubits=n,
                                                            exponent=-0.5))
    yield from reg_family(
        "grad^t", lambda n: cirq.PhaseGradientGate(
            num_qubits=n, exponent=2**(n - 1) * sympy.Symbol('t')))
    yield from reg_family(
        "grad^-t", lambda n: cirq.PhaseGradientGate(
            num_qubits=n, exponent=-2**(n - 1) * sympy.Symbol('t')))

    yield from all_qubit_permutation_cells()