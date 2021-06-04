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
import cirq.contrib.graph_device.graph_device as ccgdgd


def test_fixed_duration_undirected_graph_device_edge_eq():
    e = ccgd.FixedDurationUndirectedGraphDeviceEdge(cirq.Duration(picos=4))
    f = ccgd.FixedDurationUndirectedGraphDeviceEdge(cirq.Duration(picos=4))
    g = ccgd.FixedDurationUndirectedGraphDeviceEdge(cirq.Duration(picos=5))
    assert e == f
    assert e != g
    assert e != 4


def test_unconstrained_undirected_graph_device_edge_eq():
    e = ccgdgd._UnconstrainedUndirectedGraphDeviceEdge()
    f = ccgd.UnconstrainedUndirectedGraphDeviceEdge
    assert e == f
    assert e != 3


def test_is_undirected_device_graph():
    assert not ccgd.is_undirected_device_graph('abc')
    graph = ccgd.UndirectedHypergraph()
    assert ccgd.is_undirected_device_graph(graph)
    a, b, c, d, e = cirq.LineQubit.range(5)
    graph.add_edge((a, b))
    assert ccgd.is_undirected_device_graph(graph)
    graph.add_edge((b, c), ccgd.UnconstrainedUndirectedGraphDeviceEdge)
    assert ccgd.is_undirected_device_graph(graph)
    graph.add_edge((d, e), 'abc')
    assert not ccgd.is_undirected_device_graph(graph)
    graph = ccgd.UndirectedHypergraph(vertices=(0, 1))
    assert not ccgd.is_undirected_device_graph(graph)


def test_is_crosstalk_graph():
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    assert not ccgd.is_crosstalk_graph('abc')
    graph = ccgd.UndirectedHypergraph()
    graph.add_vertex('abc')
    assert not ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((a, b)), frozenset((c, d))), 'abc')
    assert not ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((a, b)), frozenset((c, d))), None)
    graph.add_edge((frozenset((e, f)), frozenset((c, d))), lambda _: None)
    assert ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((a, b)), frozenset((c, d))), 'abc')
    assert not ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((a, b)),), None)
    assert not ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((0, 1)), frozenset((2, 3))), None)
    assert not ccgd.is_crosstalk_graph(graph)


def test_unconstrained_undirected_graph_device_edge():
    edge = ccgd.UnconstrainedUndirectedGraphDeviceEdge
    qubits = cirq.LineQubit.range(2)
    assert edge.duration_of(cirq.X(qubits[0])) == cirq.Duration(picos=0)
    assert edge.duration_of(cirq.CZ(*qubits[:2])) == cirq.Duration(picos=0)


def test_graph_device():
    one_qubit_duration = cirq.Duration(picos=10)
    two_qubit_duration = cirq.Duration(picos=1)
    one_qubit_edge = ccgd.FixedDurationUndirectedGraphDeviceEdge(one_qubit_duration)
    two_qubit_edge = ccgd.FixedDurationUndirectedGraphDeviceEdge(two_qubit_duration)

    empty_device = ccgd.UndirectedGraphDevice()
    assert not empty_device.qubits
    assert not empty_device.edges

    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    edges = {
        (cirq.LineQubit(i), cirq.LineQubit((i + 1) % n_qubits)): two_qubit_edge
        for i in range(n_qubits)
    }
    edges.update({(cirq.LineQubit(i),): one_qubit_edge for i in range(n_qubits)})
    device_graph = ccgd.UndirectedHypergraph(labelled_edges=edges)

    def not_cnots(first_op, second_op):
        if all(
            isinstance(op, cirq.GateOperation) and op.gate == cirq.CNOT
            for op in (first_op, second_op)
        ):
            raise ValueError('Simultaneous CNOTs')

    assert ccgd.is_undirected_device_graph(device_graph)
    with pytest.raises(TypeError):
        ccgd.UndirectedGraphDevice('abc')
    constraint_edges = {
        (frozenset(cirq.LineQubit.range(2)), frozenset(cirq.LineQubit.range(2, 4))): None,
        (
            frozenset(cirq.LineQubit.range(1, 3)),
            frozenset((cirq.LineQubit(0), cirq.LineQubit(3))),
        ): not_cnots,
    }
    crosstalk_graph = ccgd.UndirectedHypergraph(labelled_edges=constraint_edges)
    assert ccgd.is_crosstalk_graph(crosstalk_graph)

    with pytest.raises(TypeError):
        ccgd.UndirectedGraphDevice(device_graph, crosstalk_graph='abc')

    graph_device = ccgd.UndirectedGraphDevice(device_graph)
    assert graph_device.crosstalk_graph == ccgd.UndirectedHypergraph()

    graph_device = ccgd.UndirectedGraphDevice(device_graph, crosstalk_graph=crosstalk_graph)
    assert sorted(graph_device.edges) == sorted(device_graph.edges)
    assert graph_device.qubits == tuple(qubits)
    assert graph_device.device_graph == device_graph
    assert graph_device.labelled_edges == device_graph.labelled_edges

    assert graph_device.duration_of(cirq.X(qubits[2])) == one_qubit_duration
    assert graph_device.duration_of(cirq.CNOT(*qubits[:2])) == two_qubit_duration
    with pytest.raises(KeyError):
        graph_device.duration_of(cirq.CNOT(qubits[0], qubits[2]))
    with pytest.raises(ValueError):
        graph_device.validate_operation(cirq.CNOT(qubits[0], qubits[2]))
    with pytest.raises(AttributeError):
        graph_device.validate_operation(list((2, 3)))

    moment = cirq.Moment([cirq.CNOT(*qubits[:2]), cirq.CNOT(*qubits[2:])])
    with pytest.raises(ValueError):
        graph_device.validate_moment(moment)

    moment = cirq.Moment([cirq.CNOT(qubits[0], qubits[3]), cirq.CZ(qubits[1], qubits[2])])
    graph_device.validate_moment(moment)

    moment = cirq.Moment([cirq.CNOT(qubits[0], qubits[3]), cirq.CNOT(qubits[1], qubits[2])])
    with pytest.raises(ValueError):
        graph_device.validate_moment(moment)


def test_graph_device_copy_and_add():
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    device_graph = ccgd.UndirectedHypergraph(labelled_edges={(a, b): None, (c, d): None})
    crosstalk_graph = ccgd.UndirectedHypergraph(
        labelled_edges={(frozenset((a, b)), frozenset((c, d))): None}
    )
    device = ccgd.UndirectedGraphDevice(device_graph=device_graph, crosstalk_graph=crosstalk_graph)
    device_graph_addend = ccgd.UndirectedHypergraph(labelled_edges={(a, b): None, (e, f): None})
    crosstalk_graph_addend = ccgd.UndirectedHypergraph(
        labelled_edges={(frozenset((a, b)), frozenset((e, f))): None}
    )
    device_addend = ccgd.UndirectedGraphDevice(
        device_graph=device_graph_addend, crosstalk_graph=crosstalk_graph_addend
    )
    device_sum = device + device_addend
    device_copy = device.__copy__()
    device_copy += device_addend
    assert device != device_copy
    assert device_copy == device_sum


def test_qubit_set():
    a, b, c, d = cirq.LineQubit.range(4)
    device_graph = ccgd.UndirectedHypergraph(labelled_edges={(a, b): None, (c, d): None})
    device = ccgd.UndirectedGraphDevice(device_graph=device_graph)
    assert device.qubit_set() == {a, b, c, d}


def test_qid_pairs():
    a, b, c, d = cirq.LineQubit.range(4)
    device_graph = ccgd.UndirectedHypergraph(labelled_edges={(a, b): None, (c, d): None})
    device = ccgd.UndirectedGraphDevice(device_graph=device_graph)
    assert len(device.qid_pairs()) == 2
