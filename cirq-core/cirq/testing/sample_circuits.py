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
from typing import TYPE_CHECKING

from cirq import _compat, ops, circuits, devices

if TYPE_CHECKING:
    import cirq


@_compat.deprecated_parameter(
    deadline='v0.15',
    fix='The returned circuit will no longer include a device object.',
    parameter_desc='device',
    match=lambda args, kwargs: 'device' in kwargs or len(args) == 4,
)
def nonoptimal_toffoli_circuit(
    q0: 'cirq.Qid',
    q1: 'cirq.Qid',
    q2: 'cirq.Qid',
    device: devices.Device = devices.UNCONSTRAINED_DEVICE,
) -> circuits.Circuit:
    ret = circuits.Circuit(
        ops.Y(q2) ** 0.5,
        ops.X(q2),
        ops.CNOT(q1, q2),
        ops.Z(q2) ** -0.25,
        ops.CNOT(q1, q2),
        ops.CNOT(q2, q1),
        ops.CNOT(q1, q2),
        ops.CNOT(q0, q1),
        ops.CNOT(q1, q2),
        ops.CNOT(q2, q1),
        ops.CNOT(q1, q2),
        ops.Z(q2) ** 0.25,
        ops.CNOT(q1, q2),
        ops.Z(q2) ** -0.25,
        ops.CNOT(q1, q2),
        ops.CNOT(q2, q1),
        ops.CNOT(q1, q2),
        ops.CNOT(q0, q1),
        ops.CNOT(q1, q2),
        ops.CNOT(q2, q1),
        ops.CNOT(q1, q2),
        ops.Z(q2) ** 0.25,
        ops.Z(q1) ** 0.25,
        ops.CNOT(q0, q1),
        ops.Z(q0) ** 0.25,
        ops.Z(q1) ** -0.25,
        ops.CNOT(q0, q1),
        ops.Y(q2) ** 0.5,
        ops.X(q2),
    )
    ret._device = device
    return ret
