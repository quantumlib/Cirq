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
from cirq.devices.noise_utils import OpIdentifier


def test_op_identifier():
    op_id = OpIdentifier(cirq.XPowGate)
    assert cirq.X(cirq.LineQubit(1)) in op_id
    assert cirq.Rx(rads=1) in op_id


def test_op_identifier_subtypes():
    gate_id = OpIdentifier(cirq.Gate)
    xpow_id = OpIdentifier(cirq.XPowGate)
    x_on_q0_id = OpIdentifier(cirq.XPowGate, cirq.LineQubit(0))
    assert xpow_id.is_proper_subtype_of(gate_id)
    assert x_on_q0_id.is_proper_subtype_of(xpow_id)
    assert x_on_q0_id.is_proper_subtype_of(gate_id)
    assert not xpow_id.is_proper_subtype_of(xpow_id)


def test_op_id_str():
    op_id = OpIdentifier(cirq.XPowGate, cirq.LineQubit(0))
    assert str(op_id) == "<class 'cirq.ops.common_gates.XPowGate'>(cirq.LineQubit(0),)"
    assert repr(op_id) == (
        "cirq.devices.noise_utils.OpIdentifier(cirq.ops.common_gates.XPowGate, cirq.LineQubit(0))"
    )


def test_op_id_swap():
    q0, q1 = cirq.LineQubit.range(2)
    base_id = OpIdentifier(cirq.CZPowGate, q0, q1)
    swap_id = OpIdentifier(base_id.gate_type, *base_id.qubits[::-1])
    assert cirq.CZ(q0, q1) in base_id
    assert cirq.CZ(q0, q1) not in swap_id
    assert cirq.CZ(q1, q0) not in base_id
    assert cirq.CZ(q1, q0) in swap_id


def test_op_id_instance():
    q0 = cirq.LineQubit.range(1)[0]
    gate = cirq.SingleQubitCliffordGate.from_xz_map((cirq.X, False), (cirq.Z, False))
    op_id = OpIdentifier(gate, q0)
    cirq.testing.assert_equivalent_repr(op_id)
