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

from cirq import linalg
from cirq.ops import gate_features, common_gates, raw_types, op_tree


class _CCZGate(gate_features.ThreeQubitGate,
               gate_features.TextDiagrammable,
               gate_features.CompositeGate,
               gate_features.KnownMatrix,
               gate_features.InterchangeableQubitsGate,
               gate_features.QasmConvertibleGate):
    """A doubly-controlled-Z."""

    def default_decompose(self, qubits):
        """An adjacency-respecting decomposition.

        0: ───T───@──────────────@───────@──────────@──────────
                  │              │       │          │
        1: ───T───X───@───T^-1───X───@───X──────@───X──────@───
                      │              │          │          │
        2: ───T───────X───T──────────X───T^-1───X───T^-1───X───
        """
        a, b, c = qubits

        # Hacky magic: avoid the non-adjacent edge.
        if hasattr(b, 'is_adjacent'):
            if not b.is_adjacent(a):
                b, c = c, b
            elif not b.is_adjacent(c):
                a, b = b, a

        t = common_gates.T
        sweep_abc = [common_gates.CNOT(a, b),
                     common_gates.CNOT(b, c)]

        yield t(a), t(b), t(c)
        yield sweep_abc
        yield t(b)**-1, t(c)
        yield sweep_abc
        yield t(c)**-1
        yield sweep_abc
        yield t(c)**-1
        yield sweep_abc

    def matrix(self):
        return np.diag([1, 1, 1, 1, 1, 1, 1, -1])

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(('@', '@', '@'))

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        lines = [
            args.format('h {0};\n', qubits[2]),
            args.format('ccx {0},{1},{2};\n', qubits[0], qubits[1], qubits[2]),
            args.format('h {0};\n', qubits[2])]
        return ''.join(lines)

    def __repr__(self) -> str:
        return 'CCZ'


class _CCXGate(gate_features.ThreeQubitGate,
               gate_features.TextDiagrammable,
               gate_features.CompositeGate,
               gate_features.KnownMatrix,
               gate_features.InterchangeableQubitsGate,
               gate_features.QasmConvertibleGate):
    """A doubly-controlled-NOT. The Toffoli gate."""

    def qubit_index_to_equivalence_group_key(self, index):
        return 0 if index < 2 else 1

    def default_decompose(self, qubits):
        c1, c2, t = qubits
        yield common_gates.H(t)
        yield CCZ(c1, c2, t)
        yield common_gates.H(t)

    def matrix(self):
        return linalg.block_diag(np.diag([1, 1, 1, 1, 1, 1]),
                                 np.array([[0, 1], [1, 0]]))

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(('@', '@', 'X'))

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        return args.format('ccx {0},{1},{2};\n',
                           qubits[0], qubits[1], qubits[2])

    def __repr__(self) -> str:
        return 'TOFFOLI'


class _CSwapGate(gate_features.ThreeQubitGate,
                 gate_features.TextDiagrammable,
                 gate_features.CompositeGate,
                 gate_features.KnownMatrix,
                 gate_features.InterchangeableQubitsGate,
                 gate_features.QasmConvertibleGate):
    """A controlled swap gate. The Fredkin gate."""

    def qubit_index_to_equivalence_group_key(self, index):
        return 0 if index == 0 else 1

    def default_decompose(self, qubits):
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

    def matrix(self):
        return linalg.block_diag(np.diag([1, 1, 1, 1, 1]),
                                 np.array([[0, 1], [1, 0]]),
                                 np.diag([1]))

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        if not args.use_unicode_characters:
            return gate_features.TextDiagramInfo(('@', 'swap', 'swap'))
        return gate_features.TextDiagramInfo(('@', '×', '×'))

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        return args.format('cswap {0},{1},{2};\n',
                           qubits[0], qubits[1], qubits[2])

    def __repr__(self) -> str:
        return 'FREDKIN'


# Explicit names.
CCZ = _CCZGate()
CCX = _CCXGate()
CSWAP = _CSwapGate()

# Common names.
TOFFOLI = CCX
FREDKIN = CSWAP
