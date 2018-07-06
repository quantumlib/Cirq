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
from cirq.line import LineQubit
from cirq.line.placement import anneal
from cirq.line.placement import greedy
from cirq.line.placement import line
from cirq.line.placement.sequence import (
    LinePlacement,
    LineSequence
)
from cirq.testing.mock import mock
from cirq.value import Duration


def test_anneal_method_calls_anneal_search():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01, q02])
    length = 2
    method = anneal.AnnealSequenceSearchStrategy

    with mock.patch.object(method, 'place_line') as place_line:
        sequences = [[q00, q01]]
        place_line.return_value = sequences

        assert line.line_placement_on_device(device, length,
                                             method) == sequences
        place_line.assert_called_once_with(device, length)


def test_greedy_method_calls_greedy_search():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01, q02])
    length = 2
    method = greedy.GreedySequenceSearchStrategy()

    with mock.patch.object(method, 'place_line') as place_line:
        sequences = [[q00, q01]]
        place_line.return_value = sequences

        assert line.line_placement_on_device(device, length,
                                             method) == sequences
        place_line.assert_called_once_with(device, length)


@mock.patch('cirq.line.placement.line.line_placement_on_device')
def test_line_on_device_calls_line_placement_on_device(
        line_placement_on_device):
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q02 = XmonQubit(0, 2)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01, q02])
    length = 2
    offset = 1
    method = greedy.GreedySequenceSearchMethod
    sequence = LineSequence([q00, q01])
    placement = LinePlacement(0, [sequence])
    line_placement_on_device.return_value = placement

    assert line.line_on_device(device, length, offset, method)[
               0] == sequence.line

    line_placement_on_device.assert_called_once_with(device, length, method)


@mock.patch('cirq.line.placement.line.line_placement_on_device')
def test_line_on_device_creates_mapping(line_placement_on_device):
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01])
    length = 2
    sequence = LineSequence([q00, q01])
    placement = LinePlacement(0, [sequence])
    line_placement_on_device.return_value = placement

    _, mapping = line.line_on_device(device, length)

    qubits = LineQubit.range(2)
    assert mapping(qubits[0]) == q00
    assert mapping(qubits[1]) == q01


@mock.patch('cirq.line.placement.line.line_placement_on_device')
def test_line_on_device_creates_mapping_with_offset(line_placement_on_device):
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01])
    length = 2
    offset = 1
    sequence = LineSequence([q00, q01])
    placement = LinePlacement(0, [sequence])
    line_placement_on_device.return_value = placement

    _, mapping = line.line_on_device(device, length, offset)

    qubits = LineQubit.range(1, 3)
    assert mapping(qubits[0]) == q00
    assert mapping(qubits[1]) == q01
