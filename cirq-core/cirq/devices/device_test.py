# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest
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


def test_qid_pairs():
    class RawDevice(cirq.Device):
        pass

    with cirq.testing.assert_deprecated('device.metadata', deadline='v0.15', count=1):
        assert RawDevice().qid_pairs() is None

    class QubitFieldDevice(cirq.Device):
        def __init__(self, qubits):
            self.qubits = qubits

    with cirq.testing.assert_deprecated('device.metadata', deadline='v0.15', count=3):

        assert len(QubitFieldDevice(cirq.LineQubit.range(10)).qid_pairs()) == 9
        assert len(QubitFieldDevice(cirq.GridQubit.rect(10, 10)).qid_pairs()) == 180
        assert len(QubitFieldDevice([cirq.NamedQubit(str(s)) for s in range(10)]).qid_pairs()) == 45


def test_decompose_operation_deprecated():
    q0 = cirq.GridQubit(0, 0)

    class RawDevice(cirq.Device):
        pass

    with cirq.testing.assert_deprecated('decompose', deadline='v0.15'):
        RawDevice().decompose_operation(cirq.H(q0))


def test_qid_pair_deprecated():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    with cirq.testing.assert_deprecated('device.metadata', deadline='v0.15', count=3):
        e1 = cirq.SymmetricalQidPair(q0, q1)
        e2 = cirq.SymmetricalQidPair(q1, q0)
        e3 = cirq.SymmetricalQidPair(q2, q3)
    assert e1 == e2
    assert e2 != e3
    assert repr(e1) == "cirq.QidPair(cirq.LineQubit(0), cirq.LineQubit(1))"

    assert len(e1) == 2
    a, b = e1
    assert (a, b) == (q0, q1)
    a, b = e2
    assert (a, b) == (q0, q1)

    assert q0 in e1
    assert q1 in e1
    assert q2 not in e1

    set1 = frozenset([e1, e2])
    set2 = frozenset([e2, e3])
    assert len(set1) == 1
    assert len(set2) == 2

    with cirq.testing.assert_deprecated('device.metadata', deadline='v0.15', count=1):
        with pytest.raises(ValueError, match='A QidPair cannot have identical qids.'):
            cirq.SymmetricalQidPair(q0, q0)


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
