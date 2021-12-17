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
import cirq.contrib.graph_device as ccgd


def test_empty_uniform_undirected_linear_device():
    n_qubits = 4
    edge_labels = {}
    device = ccgd.uniform_undirected_linear_device(n_qubits, edge_labels)
    assert device.qubits == tuple()
    assert device.edges == tuple()


def test_negative_arity_arg_uniform_undirected_linear_device():
    with pytest.raises(ValueError):
        ccgd.uniform_undirected_linear_device(5, {-1: None})
    with pytest.raises(ValueError):
        ccgd.uniform_undirected_linear_device(5, {0: None})


@pytest.mark.parametrize('arity', range(1, 5))
def test_regular_uniform_undirected_linear_device(arity):
    n_qubits = 10
    edge_labels = {arity: None}
    device = ccgd.uniform_undirected_linear_device(n_qubits, edge_labels)

    assert device.qubits == tuple(cirq.LineQubit.range(n_qubits))
    assert len(device.edges) == n_qubits - arity + 1
    for edge, label in device.labelled_edges.items():
        assert label is None
        assert len(edge) == arity
