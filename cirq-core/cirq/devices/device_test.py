# pylint: disable=wrong-or-nonexistent-copyright-notice
import networkx as nx
import cirq


def test_qubit_set_deprecated():
    class RawDevice(cirq.Device):
        pass

    with cirq.testing.assert_deprecated('qubit_set', deadline='v0.15'):
        assert RawDevice().qubit_set() is None

    class QubitFieldDevice(cirq.Device):
        def __init__(self):
            self.qubits = cirq.LineQubit.range(3)

    with cirq.testing.assert_deprecated('qubit_set', deadline='v0.15'):
        assert QubitFieldDevice().qubit_set() == frozenset(cirq.LineQubit.range(3))

    class PrivateQubitFieldDevice(cirq.Device):
        def __init__(self):
            self._qubits = cirq.LineQubit.range(4)

    with cirq.testing.assert_deprecated('qubit_set', deadline='v0.15'):
        assert PrivateQubitFieldDevice().qubit_set() == frozenset(cirq.LineQubit.range(4))

    class QubitMethodDevice(cirq.Device):
        def qubits(self):
            return cirq.LineQubit.range(5)

    with cirq.testing.assert_deprecated('qubit_set', deadline='v0.15'):
        assert QubitMethodDevice().qubit_set() == frozenset(cirq.LineQubit.range(5))

    class PrivateQubitMethodDevice(cirq.Device):
        def _qubits(self):
            return cirq.LineQubit.range(6)

    with cirq.testing.assert_deprecated('qubit_set', deadline='v0.15'):
        assert PrivateQubitMethodDevice().qubit_set() == frozenset(cirq.LineQubit.range(6))


def test_decompose_operation_deprecated():
    q0 = cirq.GridQubit(0, 0)

    class RawDevice(cirq.Device):
        pass

    with cirq.testing.assert_deprecated('decompose', deadline='v0.15'):
        RawDevice().decompose_operation(cirq.H(q0))


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
