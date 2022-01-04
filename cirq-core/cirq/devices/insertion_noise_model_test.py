# Copyright 2021 The Cirq Developers
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

import cirq
from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.devices.noise_utils import (
    PHYSICAL_GATE_TAG,
    OpIdentifier,
)


def test_insertion_noise():
    q0, q1 = cirq.LineQubit.range(2)
    op_id0 = OpIdentifier(cirq.XPowGate, q0)
    op_id1 = OpIdentifier(cirq.ZPowGate, q1)
    model = InsertionNoiseModel(
        {op_id0: cirq.T(q0), op_id1: cirq.H(q1)}, require_physical_tag=False
    )
    assert not model.prepend

    moment_0 = cirq.Moment(cirq.X(q0), cirq.X(q1))
    assert model.noisy_moment(moment_0, system_qubits=[q0, q1]) == [
        moment_0,
        cirq.Moment(cirq.T(q0)),
    ]

    moment_1 = cirq.Moment(cirq.Z(q0), cirq.Z(q1))
    assert model.noisy_moment(moment_1, system_qubits=[q0, q1]) == [
        moment_1,
        cirq.Moment(cirq.H(q1)),
    ]

    moment_2 = cirq.Moment(cirq.X(q0), cirq.Z(q1))
    assert model.noisy_moment(moment_2, system_qubits=[q0, q1]) == [
        moment_2,
        cirq.Moment(cirq.T(q0), cirq.H(q1)),
    ]

    moment_3 = cirq.Moment(cirq.Z(q0), cirq.X(q1))
    assert model.noisy_moment(moment_3, system_qubits=[q0, q1]) == [moment_3]


def test_colliding_noise_qubits():
    # Check that noise affecting other qubits doesn't cause issues.
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    op_id0 = OpIdentifier(cirq.CZPowGate)
    model = InsertionNoiseModel({op_id0: cirq.CNOT(q1, q2)}, require_physical_tag=False)

    moment_0 = cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3))
    assert model.noisy_moment(moment_0, system_qubits=[q0, q1, q2, q3]) == [
        moment_0,
        cirq.Moment(cirq.CNOT(q1, q2)),
        cirq.Moment(cirq.CNOT(q1, q2)),
    ]


def test_prepend():
    q0, q1 = cirq.LineQubit.range(2)
    op_id0 = OpIdentifier(cirq.XPowGate, q0)
    op_id1 = OpIdentifier(cirq.ZPowGate, q1)
    model = InsertionNoiseModel(
        {op_id0: cirq.T(q0), op_id1: cirq.H(q1)}, prepend=True, require_physical_tag=False
    )

    moment_0 = cirq.Moment(cirq.X(q0), cirq.Z(q1))
    assert model.noisy_moment(moment_0, system_qubits=[q0, q1]) == [
        cirq.Moment(cirq.T(q0), cirq.H(q1)),
        moment_0,
    ]


def test_require_physical_tag():
    q0, q1 = cirq.LineQubit.range(2)
    op_id0 = OpIdentifier(cirq.XPowGate, q0)
    op_id1 = OpIdentifier(cirq.ZPowGate, q1)
    model = InsertionNoiseModel({op_id0: cirq.T(q0), op_id1: cirq.H(q1)})
    assert model.require_physical_tag

    moment_0 = cirq.Moment(cirq.X(q0).with_tags(PHYSICAL_GATE_TAG), cirq.Z(q1))
    assert model.noisy_moment(moment_0, system_qubits=[q0, q1]) == [
        moment_0,
        cirq.Moment(cirq.T(q0)),
    ]


def test_supertype_matching():
    # Demonstrate that the model applies the closest matching type
    # if multiple types match a given gate.
    q0 = cirq.LineQubit(0)
    op_id0 = OpIdentifier(cirq.Gate, q0)
    op_id1 = OpIdentifier(cirq.XPowGate, q0)
    model = InsertionNoiseModel(
        {op_id0: cirq.T(q0), op_id1: cirq.S(q0)}, require_physical_tag=False
    )

    moment_0 = cirq.Moment(cirq.Rx(rads=1).on(q0))
    assert model.noisy_moment(moment_0, system_qubits=[q0]) == [
        moment_0,
        cirq.Moment(cirq.S(q0)),
    ]

    moment_1 = cirq.Moment(cirq.Y(q0))
    assert model.noisy_moment(moment_1, system_qubits=[q0]) == [
        moment_1,
        cirq.Moment(cirq.T(q0)),
    ]
