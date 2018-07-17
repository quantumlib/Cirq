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
from cirq.ops import gate_features, common_gates, raw_types


class _CCZGate(gate_features.ThreeQubitGate,
               gate_features.TextDiagrammable,
               gate_features.CompositeGate,
               gate_features.KnownMatrix,
               gate_features.InterchangeableQubitsGate,
               gate_features.QasmConvertableGate):
    """A doubly-controlled-Z."""

    def default_decompose(self, qubits):
        a, b, c = qubits

        # Hacky magic: avoid the non-adjacent edge.
        if hasattr(b, 'is_adjacent'):
            if not b.is_adjacent(a):
                b, c = c, b
            elif not b.is_adjacent(c):
                a, b = b, a

        yield common_gates.T.on_each([a, b, c])  # Phase a, b, and c.
        yield common_gates.CNOT(c, b)
        yield common_gates.T(b)**-1  # Counter-phase b ⊕ c.
        yield common_gates.CNOT(a, b)
        yield common_gates.T(b)  # Phase a ⊕ b ⊕ c.
        yield common_gates.CNOT(c, b)
        yield common_gates.T(b)**-1  # Counter-phase a ⊕ b.
        yield common_gates.CNOT(a, b)

        # If a then toggle b and c.
        yield common_gates.CNOT(b, c)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)

        yield common_gates.T(c)**-1  # Counter-phase a ⊕ c.

        # If a then un-toggle b and c.
        yield common_gates.CNOT(b, c)
        yield common_gates.CNOT(a, b)
        yield common_gates.CNOT(b, c)

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
            args.format('h {};\n', qubits[2]),
            args.format('ccx {},{},{};\n', qubits[0], qubits[1], qubits[2]),
            args.format('h {};\n', qubits[2])]
        return ''.join(lines)

    def __repr__(self) -> str:
        return 'CCZ'


class _CCXGate(gate_features.ThreeQubitGate,
               gate_features.TextDiagrammable,
               gate_features.CompositeGate,
               gate_features.KnownMatrix,
               gate_features.InterchangeableQubitsGate,
               gate_features.QasmConvertableGate):
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
        return args.format('ccx {},{},{};\n', qubits[0], qubits[1], qubits[2])

    def __repr__(self) -> str:
        return 'TOFFOLI'


class _CSwapGate(gate_features.ThreeQubitGate,
                 gate_features.TextDiagrammable,
                 gate_features.CompositeGate,
                 gate_features.KnownMatrix,
                 gate_features.InterchangeableQubitsGate,
                 gate_features.QasmConvertableGate):
    """A controlled swap gate. The Fredkin gate."""

    def qubit_index_to_equivalence_group_key(self, index):
        return 0 if index == 0 else 1

    def default_decompose(self, qubits):
        c, t1, t2 = qubits

        # Hacky magic: cross the non-adjacent edge.
        need_hop = hasattr(t1, 'is_adjacent') and not t1.is_adjacent(t2)

        cnot = common_gates.CNOT(t2, t1) if not need_hop else [
            common_gates.CNOT(t2, c),
            common_gates.CNOT(c, t1),
            common_gates.CNOT(t2, c),
            common_gates.CNOT(c, t1),
        ]
        yield cnot
        yield TOFFOLI(c, t1, t2)
        yield cnot

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
        return args.format('cswap {},{},{};\n', qubits[0], qubits[1], qubits[2])

    def __repr__(self) -> str:
        return 'FREDKIN'


# Explicit names.
CCZ = _CCZGate()
CCX = _CCXGate()
CSWAP = _CSwapGate()

# Common names.
TOFFOLI = CCX
FREDKIN = CSWAP
