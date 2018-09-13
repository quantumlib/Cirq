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

from cirq.contrib.acquaintance.gates import (
        AcquaintanceOpportunityGate, SwapNetworkGate)
from cirq.contrib.acquaintance.devices import (
        UnconstrainedAcquaintanceDevice,
        get_acquaintance_size)


def test_acquaintance_device():
    with pytest.raises(ValueError):
        op = cirq.X(cirq.NamedQubit('q'))
        UnconstrainedAcquaintanceDevice.validate_operation(op)

    qubits = cirq.LineQubit.range(4)
    ACQUAINT = AcquaintanceOpportunityGate()
    swap_network = SwapNetworkGate((1, 2, 1))
    UnconstrainedAcquaintanceDevice.validate_operation(
        ACQUAINT(*qubits[:2]))
    UnconstrainedAcquaintanceDevice.validate_operation(
        swap_network(*qubits))


def test_get_acquaintance_size():
    with pytest.raises(TypeError):
        get_acquaintance_size(cirq.Circuit())

    with pytest.raises(TypeError):
        get_acquaintance_size(3)

    circuit = cirq.Circuit(device=UnconstrainedAcquaintanceDevice)
    get_acquaintance_size(circuit)
