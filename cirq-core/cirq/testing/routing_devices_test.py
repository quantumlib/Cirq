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
import pytest
import cirq
import networkx as nx

from cirq.testing.routing_devices import construct_grid_device


def test_grid_device():
    device = construct_grid_device(5)
    device_graph = device.metadata.nx_graph

    assert all(q in device_graph.nodes for q in cirq.GridQubit.square(5))
    assert nx.is_isomorphic(nx.grid_2d_graph(5, 5), device_graph)


def test_op_validation():
    device = construct_grid_device(5)

    with pytest.raises(ValueError, match="Qubit not on device"):
        device.validate_operation(cirq.X(cirq.NamedQubit("a")))
    with pytest.raises(ValueError, match="Qubit not on device"):
        device.validate_operation(cirq.CNOT(cirq.NamedQubit("a"), cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError, match="Qubit not on device"):
        device.validate_operation(cirq.CNOT(cirq.GridQubit(5, 4), cirq.GridQubit(4, 4)))

    with pytest.raises(ValueError, match="Qubit pair is not valid on device"):
        device.validate_operation(cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(0, 2)))
    with pytest.raises(ValueError, match="Qubit pair is not valid on device"):
        device.validate_operation(cirq.CNOT(cirq.GridQubit(2, 0), cirq.GridQubit(0, 0)))

    device.validate_operation(cirq.CNOT(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    device.validate_operation(cirq.CNOT(cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)))
