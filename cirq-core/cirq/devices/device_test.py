# pylint: disable=wrong-or-nonexistent-copyright-notice
import networkx as nx

import cirq


def test_device_metadata():
    class RawDevice(cirq.Device):
        pass

    assert RawDevice().metadata is None


def test_metadata():
    qubits = cirq.LineQubit.range(4)
    graph = nx.star_graph(3)
    metadata = cirq.DeviceMetadata(qubits, graph)
    assert metadata.qubit_set == frozenset(qubits)
    assert metadata.nx_graph == graph


def test_metadata_json_load_logic():
    qubits = cirq.LineQubit.range(4)
    graph = nx.star_graph(3)
    metadata = cirq.DeviceMetadata(qubits, graph)
    str_rep = cirq.to_json(metadata)
    assert metadata == cirq.read_json(json_text=str_rep)


def test_metadata_equality():
    qubits = cirq.LineQubit.range(4)
    graph = nx.star_graph(3)
    graph2 = nx.star_graph(3)
    graph.add_edge(1, 2, directed=False)
    graph2.add_edge(1, 2, directed=True)

    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.DeviceMetadata(qubits, graph))
    eq.add_equality_group(cirq.DeviceMetadata(qubits, graph2))
    eq.add_equality_group(cirq.DeviceMetadata(qubits[1:], graph))
