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
import cirq.contrib.acquaintance as cca


def test_acquaintance_device():
    with pytest.raises(ValueError):
        op = cirq.X(cirq.NamedQubit('q'))
        cca.UnconstrainedAcquaintanceDevice.validate_operation(op)

    qubits = cirq.LineQubit.range(4)
    swap_network = cca.SwapNetworkGate((1, 2, 1))
    cca.UnconstrainedAcquaintanceDevice.validate_operation(cca.acquaint(*qubits[:2]))
    cca.UnconstrainedAcquaintanceDevice.validate_operation(swap_network(*qubits))


def test_not_operation():
    with pytest.raises(TypeError):
        _ = cca.get_acquaintance_size(cirq.LineQubit(1))
