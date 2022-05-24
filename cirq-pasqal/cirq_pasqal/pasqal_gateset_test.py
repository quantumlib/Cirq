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
#
import pytest
import cirq
import cirq_pasqal

Q, Q2, Q3 = cirq.LineQubit.range(3)


@pytest.mark.parametrize(
    "op,expected",
    [
        (cirq.H(Q), True),
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
    gs = cirq_pasqal.PasqalGateset()
    assert gs.validate(op) == expected
    assert gs.validate(cirq.Circuit(op)) == expected


@pytest.mark.parametrize(
    "op,expected",
    [
        (cirq.H(Q), True),
        (cirq.HPowGate(exponent=0.5)(Q), False),
        (cirq.PhasedXPowGate(exponent=0.25, phase_exponent=0.125)(Q), True),
        (cirq.ParallelGate(cirq.X, num_copies=3)(Q, Q2, Q3), True),
        (cirq.CZPowGate(exponent=0.5)(Q, Q2), False),
        (cirq.CZ(Q, Q2), True),
        (cirq.CNOT(Q, Q2), False),
        (cirq.CCNOT(Q, Q2, Q3), False),
        (cirq.CCZ(Q, Q2, Q3), False),
        (cirq.Z(Q).controlled_by(Q2), True),
        (cirq.X(Q).controlled_by(Q2, Q3), False),
        (cirq.Z(Q).controlled_by(Q2, Q3), False),
        (cirq.ZPowGate(exponent=0.5)(Q).controlled_by(Q2, Q3), False),
    ],
)
def test_control_gates_not_included(op: cirq.Operation, expected: bool):
    gs = cirq_pasqal.PasqalGateset(include_additional_controlled_ops=False)
    assert gs.validate(op) == expected
    assert gs.validate(cirq.Circuit(op)) == expected


@pytest.mark.parametrize(
    "op",
    [
        cirq.X(Q),
        cirq.SWAP(Q, Q2),
        cirq.ISWAP(Q, Q2),
        cirq.CCNOT(Q, Q2, Q3),
        cirq.CCZ(Q, Q2, Q3),
        cirq.ParallelGate(cirq.X, num_copies=3)(Q, Q2, Q3),
        cirq.SWAP(Q, Q2).controlled_by(Q3),
    ],
)
def test_decomposition(op: cirq.Operation):
    circuit = cirq.Circuit(op)
    gs = cirq_pasqal.PasqalGateset()
    gs2 = cirq_pasqal.PasqalGateset(include_additional_controlled_ops=False)
    for gateset in [gs, gs2]:
        decomposed_circuit = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
        for new_op in decomposed_circuit.all_operations():
            assert gs.validate(new_op)


def test_repr():
    cirq.testing.assert_equivalent_repr(
        cirq_pasqal.PasqalGateset(), setup_code='import cirq_pasqal'
    )
    cirq.testing.assert_equivalent_repr(
        cirq_pasqal.PasqalGateset(include_additional_controlled_ops=False),
        setup_code='import cirq_pasqal',
    )
