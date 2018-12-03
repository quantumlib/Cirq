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

"""Common quantum gates that target three qubits."""

from typing import Optional, Tuple

import numpy as np

from cirq import linalg, protocols
from cirq.ops import (
    common_gates,
    controlled_gate,
    eigen_gate,
    gate_features,
    op_tree,
    raw_types,
)


class CCZPowGate(eigen_gate.EigenGate,
                 gate_features.ThreeQubitGate,
                 gate_features.InterchangeableQubitsGate):
    """A doubly-controlled-Z that can be raised to a power.

    The matrix of `CCZ**t` is `diag(1, 1, 1, 1, 1, 1, 1, exp(i pi t))`.
    """

    def _eigen_components(self):
        return [
            (0, np.diag([1, 1, 1, 1, 1, 1, 1, 0])),
            (1, np.diag([0, 0, 0, 0, 0, 0, 0, 1])),
        ]

    def _decompose_(self, qubits):
        """An adjacency-respecting decomposition.

        0: ───p───@──────────────@───────@──────────@──────────
                  │              │       │          │
        1: ───p───X───@───p^-1───X───@───X──────@───X──────@───
                      │              │          │          │
        2: ───p───────X───p──────────X───p^-1───X───p^-1───X───

        where p = T**self._exponent
        """
        a, b, c = qubits

        # Hacky magic: avoid the non-adjacent edge.
        if hasattr(b, 'is_adjacent'):
            if not b.is_adjacent(a):
                b, c = c, b
            elif not b.is_adjacent(c):
                a, b = b, a

        p = common_gates.T**self._exponent
        sweep_abc = [common_gates.CNOT(a, b),
                     common_gates.CNOT(b, c)]

        yield p(a), p(b), p(c)
        yield sweep_abc
        yield p(b)**-1, p(c)
        yield sweep_abc
        yield p(c)**-1
        yield sweep_abc
        yield p(c)**-1
        yield sweep_abc

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> np.ndarray:
        if protocols.is_parameterized(self):
            return NotImplemented
        ooo = args.subspace_index(0b111)
        args.target_tensor[ooo] *= np.exp(1j * self.exponent * np.pi)
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return args.target_tensor

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            ('@', '@', '@'),
            exponent=self._diagram_exponent(args))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        if self._exponent != 1:
            return None

        args.validate_version('2.0')
        lines = [
            args.format('h {0};\n', qubits[2]),
            args.format('ccx {0},{1},{2};\n', qubits[0], qubits[1], qubits[2]),
            args.format('h {0};\n', qubits[2])]
        return ''.join(lines)

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.CCZ'
            return '(cirq.CCZ**{!r})'.format(self._exponent)
        return (
            'cirq.CCZPowGate(exponent={!r}, '
            'global_shift={!r})'
        ).format(self._exponent, self._global_shift)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CCZ'
        return 'CCZ**{}'.format(self._exponent)


class CCXPowGate(eigen_gate.EigenGate,
                 gate_features.ThreeQubitGate,
                 gate_features.InterchangeableQubitsGate):
    """A Toffoli (doubly-controlled-NOT) that can be raised to a power.

    The matrix of `CCX**t` is an 8x8 identity except the bottom right 2x2 area
    is the matrix of `X**t`.
    """

    def _eigen_components(self):
        return [
            (0, linalg.block_diag(np.diag([1, 1, 1, 1, 1, 1]),
                                  np.array([[0.5, 0.5], [0.5, 0.5]]))),
            (1, linalg.block_diag(np.diag([0, 0, 0, 0, 0, 0]),
                                  np.array([[0.5, -0.5], [-0.5, 0.5]]))),
        ]

    def qubit_index_to_equivalence_group_key(self, index):
        return index < 2

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> np.ndarray:
        if protocols.is_parameterized(self):
            return NotImplemented
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            args.target_tensor *= p
        return protocols.apply_unitary(
            controlled_gate.ControlledGate(
                controlled_gate.ControlledGate(
                    common_gates.X**self.exponent)),
            protocols.ApplyUnitaryArgs(
                args.target_tensor,
                args.available_buffer,
                args.axes),
            default=NotImplemented)

    def _decompose_(self, qubits):
        c1, c2, t = qubits
        yield common_gates.H(t)
        yield CCZ(c1, c2, t)**self._exponent
        yield common_gates.H(t)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            ('@', '@', 'X'),
            exponent=self._diagram_exponent(args))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        if self._exponent != 1:
            return None

        args.validate_version('2.0')
        return args.format('ccx {0},{1},{2};\n',
                           qubits[0], qubits[1], qubits[2])

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.TOFFOLI'
            return '(cirq.TOFFOLI**{!r})'.format(self._exponent)
        return (
            'cirq.CCXPowGate(exponent={!r}, '
            'global_shift={!r})'
        ).format(self._exponent, self._global_shift)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'TOFFOLI'
        return 'TOFFOLI**{}'.format(self._exponent)


class CSwapGate(gate_features.ThreeQubitGate,
                gate_features.InterchangeableQubitsGate):
    """A controlled swap gate. The Fredkin gate."""

    def qubit_index_to_equivalence_group_key(self, index):
        return 0 if index == 0 else 1

    def _decompose_(self, qubits):
        c, t1, t2 = qubits

        # Hacky magic: special case based on adjacency.
        if hasattr(t1, 'is_adjacent'):
            if not t1.is_adjacent(t2):
                # Targets separated by control.
                return self._decompose_inside_control(t1, c, t2)
            if not t1.is_adjacent(c):
                # Control separated from t1 by t2.
                return self._decompose_outside_control(c, t2, t1)

        return self._decompose_outside_control(c, t1, t2)

    def _decompose_inside_control(self,
                                  target1: raw_types.QubitId,
                                  control: raw_types.QubitId,
                                  target2: raw_types.QubitId
                                  ) -> op_tree.OP_TREE:
        """A decomposition assuming the control separates the targets.

        target1: ─@─X───────T──────@────────@─────────X───@─────X^-0.5─
                  │ │              │        │         │   │
        control: ─X─@─X─────@─T^-1─X─@─T────X─@─X^0.5─@─@─X─@──────────
                      │     │        │        │         │   │
        target2: ─────@─H─T─X─T──────X─T^-1───X─T^-1────X───X─H─S^-1───
        """
        a, b, c = target1, control, target2
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, a)
        yield common_gates.CNOT(c, b)
        yield common_gates.H(c)
        yield common_gates.T(c)
        yield common_gates.CNOT(b, c)
        yield common_gates.T(a)
        yield common_gates.T(b)**-1
        yield common_gates.T(c)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield common_gates.T(b)
        yield common_gates.T(c)**-1
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield common_gates.X(b)**0.5
        yield common_gates.T(c)**-1
        yield common_gates.CNOT(b, a)
        yield common_gates.CNOT(b, c)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)
        yield common_gates.H(c)
        yield common_gates.S(c)**-1
        yield common_gates.X(a)**-0.5

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs) -> np.ndarray:
        return protocols.apply_unitary(
            controlled_gate.ControlledGate(common_gates.SWAP),
            protocols.ApplyUnitaryArgs(
                args.target_tensor,
                args.available_buffer,
                args.axes),
            default=NotImplemented)

    def _decompose_outside_control(self,
                                   control: raw_types.QubitId,
                                   near_target: raw_types.QubitId,
                                   far_target: raw_types.QubitId
                                   ) -> op_tree.OP_TREE:
        """A decomposition assuming one of the targets is in the middle.

        control: ───T──────@────────@───@────────────@────────────────
                           │        │   │            │
           near: ─X─T──────X─@─T^-1─X─@─X────@─X^0.5─X─@─X^0.5────────
                  │          │        │      │         │
            far: ─@─Y^-0.5─T─X─T──────X─T^-1─X─T^-1────X─S─────X^-0.5─
        """
        a, b, c = control, near_target, far_target

        t = common_gates.T
        sweep_abc = [common_gates.CNOT(a, b),
                     common_gates.CNOT(b, c)]

        yield common_gates.CNOT(c, b)
        yield common_gates.Y(c)**-0.5
        yield t(a), t(b), t(c)
        yield sweep_abc
        yield t(b) ** -1, t(c)
        yield sweep_abc
        yield t(c) ** -1
        yield sweep_abc
        yield t(c) ** -1
        yield common_gates.X(b)**0.5
        yield sweep_abc
        yield common_gates.S(c)
        yield common_gates.X(b)**0.5
        yield common_gates.X(c)**-0.5

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return linalg.block_diag(np.diag([1, 1, 1, 1, 1]),
                                 np.array([[0, 1], [1, 0]]),
                                 np.diag([1]))

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return protocols.CircuitDiagramInfo(('@', 'swap', 'swap'))
        return protocols.CircuitDiagramInfo(('@', '×', '×'))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        args.validate_version('2.0')
        return args.format('cswap {0},{1},{2};\n',
                           qubits[0], qubits[1], qubits[2])

    def __str__(self) -> str:
        return 'FREDKIN'

    def __repr__(self) -> str:
        return 'cirq.FREDKIN'


# Explicit names.
CCZ = CCZPowGate()
CCX = CCXPowGate()
CSWAP = CSwapGate()

# Common names.
TOFFOLI = CCX
FREDKIN = CSWAP
