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

import pytest

import cirq

Q, Q2, Q3 = cirq.LineQubit.range(3)


@pytest.mark.parametrize(
    "op,expected",
    [
        (cirq.H(Q), False),
        (cirq.HPowGate(exponent=0.5)(Q), False),
        (cirq.PhasedXPowGate(exponent=0.25, phase_exponent=0.125)(Q), True),
        (cirq.XPowGate(exponent=0.5)(Q), True),
        (cirq.YPowGate(exponent=0.25)(Q), True),
        (cirq.ZPowGate(exponent=0.125)(Q), True),
        (cirq.CZPowGate(exponent=0.5)(Q, Q2), False),
        (cirq.CZ(Q, Q2), True),
        (cirq.CNOT(Q, Q2), True),
        (cirq.SWAP(Q, Q2), False),
        (cirq.ISWAP(Q, Q2), False),
        (cirq.CCNOT(Q, Q2, Q3), True),
        (cirq.CCZ(Q, Q2, Q3), True),
        (cirq.ParallelGate(cirq.X, num_copies=3)(Q, Q2, Q3), True),
        (cirq.ParallelGate(cirq.Y, num_copies=3)(Q, Q2, Q3), True),
        (cirq.ParallelGate(cirq.Z, num_copies=3)(Q, Q2, Q3), True),
        (cirq.X(Q).controlled_by(Q2, Q3), True),
        (cirq.Z(Q).controlled_by(Q2, Q3), True),
        (cirq.ZPowGate(exponent=0.5)(Q).controlled_by(Q2, Q3), False),
    ],
)
def test_gateset(op: cirq.Operation, expected: bool):
    assert cirq.is_native_neutral_atom_op(op) == expected
    if op.gate is not None:
        assert cirq.is_native_neutral_atom_gate(op.gate) == expected
