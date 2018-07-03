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

from cirq.google import XmonDevice, XmonQubit
from cirq.line.placement import anneal
from cirq.line.placement import greedy
from cirq.line.placement.line import (
    line_placement_on_device
)
from cirq.testing.mock import mock
from cirq.value import Duration


def test_anneal_method_calls_anneal_search():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q03 = XmonQubit(0, 3)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01, q03])
    length = 2
    method = anneal.AnnealSequenceSearchMethod

    with mock.patch.object(method, 'place_line') as place_line:
        sequences = [[q00, q01]]
        place_line.return_value = sequences

        assert line_placement_on_device(device, length, method) == sequences
        place_line.assert_called_once_with(device, length)


def test_greedy_method_calls_greedy_search():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q03 = XmonQubit(0, 3)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01, q03])
    length = 2
    method = greedy.GreedySequenceSearchMethod()

    with mock.patch.object(method, 'place_line') as place_line:
        sequences = [[q00, q01]]
        place_line.return_value = sequences

        assert line_placement_on_device(device, length, method) == sequences
        place_line.assert_called_once_with(device, length)
