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

from cirq.line.placement import anneal, greedy
from cirq.line.placement.place import (
    LinePlacementOptions,
    SequenceSearchMethod,
    place_on_device
)
from cirq.testing.mock import mock
from cirq.google import XmonDevice, XmonQubit
from cirq.value import Duration


@mock.patch('cirq.line.placement.anneal.anneal_sequence')
def test_anneal_method_calls_anneal_search(anneal_sequence):
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q03 = XmonQubit(0, 3)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01, q03])
    search_method = anneal.AnnealSequenceSearchMethod()
    seed = 1
    sequences = [[q00, q01]]
    anneal_sequence.return_value = sequences

    assert place_on_device(device,
                           LinePlacementOptions(search_method=search_method,
                                                seed=seed)) == sequences
    anneal_sequence.assert_called_once_with(device, search_method, seed=seed)


@mock.patch('cirq.line.placement.greedy.greedy_sequence')
def test_greedy_method_calls_greedy_search(greedy_sequence):
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    q03 = XmonQubit(0, 3)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01, q03])
    search_method = greedy.GreedySequenceSearchMethod()
    sequences = [[q00, q01]]
    greedy_sequence.return_value = sequences

    assert place_on_device(device,
                           LinePlacementOptions(
                               search_method=search_method)) == sequences
    greedy_sequence.assert_called_once_with(device, search_method)


def test_search_fails_on_unknown_method():
    q00 = XmonQubit(0, 0)
    q01 = XmonQubit(0, 1)
    device = XmonDevice(Duration(nanos=0), Duration(nanos=0),
                        Duration(nanos=0), qubits=[q00, q01])
    with pytest.raises(ValueError):
        place_on_device(device, LinePlacementOptions(
            search_method=SequenceSearchMethod()))
