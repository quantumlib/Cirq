# Copyright 2022 The Cirq Developers
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
import cirq.neutral_atoms.neutral_atom_gateset as nag

Q = cirq.LineQubit.range(10)


@pytest.mark.parametrize(
    'op,is_in_gateset',
    (
        (cirq.X.on(Q[0]), True),
        (cirq.Y.on(Q[0]), True),
        (cirq.I.on(Q[0]), True),
        (cirq.measure(*Q), True),
        (cirq.ParallelGate(cirq.X, 3).on(*Q[:3]), True),
        (cirq.ParallelGate(cirq.X, 4).on(*Q[:4]), False),
        (cirq.ParallelGate(cirq.X, 10).on(*Q), False),
        (cirq.ParallelGate(cirq.Y, 3).on(*Q[:3]), True),
        (cirq.ParallelGate(cirq.Y, 4).on(*Q[:4]), False),
        (cirq.ParallelGate(cirq.Y, 10).on(*Q), False),
        (cirq.ParallelGate(cirq.Z, 3).on(*Q[:3]), True),
        (cirq.ParallelGate(cirq.Z, 4).on(*Q[:4]), True),
        (cirq.ParallelGate(cirq.Z, 5).on(*Q[:5]), False),
        (cirq.ParallelGate(cirq.Z, 10).on(*Q), False),
        (
            cirq.ParallelGate(cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.25), 3).on(*Q[:3]),
            True,
        ),
        (
            cirq.ParallelGate(cirq.PhasedXPowGate(exponent=0.5, phase_exponent=0.25), 4).on(*Q[:4]),
            False,
        ),
        (cirq.CNOT.on(Q[0], Q[1]), True),
        ((cirq.CNOT**0.5).on(Q[0], Q[1]), False),
        (cirq.CZ.on(Q[0], Q[1]), True),
        ((cirq.CZ**0.5).on(Q[0], Q[1]), False),
        (cirq.CCZ.on(Q[0], Q[1], Q[2]), True),
        ((cirq.CCZ**0.5).on(Q[0], Q[1], Q[2]), False),
        ((cirq.TOFFOLI**0.5).on(Q[0], Q[1], Q[2]), False),
    ),
)
def test_gateset(op: cirq.Operation, is_in_gateset: bool):
    gateset = nag.NeutralAtomGateset(max_parallel_z=4, max_parallel_xy=3)
    assert gateset.validate(op) == is_in_gateset
    converted_ops = cirq.optimize_for_target_gateset(cirq.Circuit(op), gateset=gateset)
    if is_in_gateset:
        assert converted_ops == cirq.Circuit(op)
        assert gateset.validate(converted_ops)


def test_gateset_qubits():
    assert nag.NeutralAtomGateset(max_parallel_z=4, max_parallel_xy=3).num_qubits() == 2
