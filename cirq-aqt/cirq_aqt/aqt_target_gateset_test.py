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

"""Tests for AQTTargetGateset."""

import pytest
import sympy

import cirq
from cirq_aqt import aqt_target_gateset

Q, Q2, Q3, Q4 = cirq.LineQubit.range(4)


@pytest.mark.parametrize(
    "op,expected",
    [
        (cirq.H(Q), False),
        (cirq.HPowGate(exponent=0.5)(Q), False),
        (cirq.XX(Q, Q2), True),
        (cirq.measure(Q), True),
        (cirq.XPowGate(exponent=0.5)(Q), False),
        (cirq.YPowGate(exponent=0.25)(Q), False),
        (cirq.ZPowGate(exponent=0.125)(Q), True),
        (cirq.PhasedXPowGate(exponent=0.25, phase_exponent=0.125)(Q), True),
        (cirq.CZPowGate(exponent=0.5)(Q, Q2), False),
        (cirq.CZ(Q, Q2), False),
        (cirq.CNOT(Q, Q2), False),
        (cirq.SWAP(Q, Q2), False),
        (cirq.ISWAP(Q, Q2), False),
        (cirq.CCNOT(Q, Q2, Q3), False),
        (cirq.CCZ(Q, Q2, Q3), False),
        (cirq.ParallelGate(cirq.X, num_copies=3)(Q, Q2, Q3), False),
        (cirq.ParallelGate(cirq.Y, num_copies=3)(Q, Q2, Q3), False),
        (cirq.ParallelGate(cirq.Z, num_copies=3)(Q, Q2, Q3), False),
        (cirq.X(Q).controlled_by(Q2, Q3), False),
        (cirq.Z(Q).controlled_by(Q2, Q3), False),
        (cirq.ZPowGate(exponent=0.5)(Q).controlled_by(Q2, Q3), False),
    ],
)
def test_gateset(op: cirq.Operation, expected: bool):
    gs = aqt_target_gateset.AQTTargetGateset()
    assert gs.validate(op) == expected
    assert gs.validate(cirq.Circuit(op)) == expected


def test_decompose_single_qubit_operation():
    gs = aqt_target_gateset.AQTTargetGateset()
    tgoph = gs.decompose_to_target_gateset(cirq.H(Q), 0)
    assert len(tgoph) == 2
    assert isinstance(tgoph[0].gate, cirq.Rx)
    assert isinstance(tgoph[1].gate, cirq.Ry)
    tcoph = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.H(Q))).with_tags('tagged')
    tgtcoph = gs.decompose_to_target_gateset(tcoph, 0)
    assert len(tgtcoph) == 2
    assert isinstance(tgtcoph[0].gate, cirq.Rx)
    assert isinstance(tgtcoph[1].gate, cirq.Ry)
    tgopz = gs.decompose_to_target_gateset(cirq.Z(Q), 0)
    assert len(tgopz) == 1
    assert isinstance(tgopz[0].gate, cirq.ZPowGate)
    theta = sympy.Symbol('theta')
    assert gs.decompose_to_target_gateset(cirq.H(Q) ** theta, 0) is NotImplemented
    return


def test_decompose_two_qubit_operation():
    gs = aqt_target_gateset.AQTTargetGateset()
    tgopsqrtxx = gs.decompose_to_target_gateset(cirq.XX(Q, Q2) ** 0.5, 0)
    assert len(tgopsqrtxx) == 1
    assert isinstance(tgopsqrtxx[0].gate, cirq.XXPowGate)
    theta = sympy.Symbol('theta')
    assert gs.decompose_to_target_gateset(cirq.XX(Q, Q2) ** theta, 0) is NotImplemented
    return


def test_postprocess_transformers():
    gs = aqt_target_gateset.AQTTargetGateset()
    assert len(gs.postprocess_transformers) == 2
