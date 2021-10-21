# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest
import cirq


def test_qubit_set():
    class RawDevice(cirq.Device):
        pass

    assert RawDevice().qubit_set() is None

    class QubitFieldDevice(cirq.Device):
        def __init__(self):
            self.qubits = cirq.LineQubit.range(3)

    assert QubitFieldDevice().qubit_set() == frozenset(cirq.LineQubit.range(3))

    class PrivateQubitFieldDevice(cirq.Device):
        def __init__(self):
            self._qubits = cirq.LineQubit.range(4)

    assert PrivateQubitFieldDevice().qubit_set() == frozenset(cirq.LineQubit.range(4))

    class QubitMethodDevice(cirq.Device):
        def qubits(self):
            return cirq.LineQubit.range(5)

    assert QubitMethodDevice().qubit_set() == frozenset(cirq.LineQubit.range(5))

    class PrivateQubitMethodDevice(cirq.Device):
        def _qubits(self):
            return cirq.LineQubit.range(6)

    assert PrivateQubitMethodDevice().qubit_set() == frozenset(cirq.LineQubit.range(6))


def test_qid_pairs():
    class RawDevice(cirq.Device):
        pass

    assert RawDevice().qid_pairs() is None

    class QubitFieldDevice(cirq.Device):
        def __init__(self, qubits):
            self.qubits = qubits

    assert len(QubitFieldDevice(cirq.LineQubit.range(10)).qid_pairs()) == 9
    assert len(QubitFieldDevice(cirq.GridQubit.rect(10, 10)).qid_pairs()) == 180
    assert len(QubitFieldDevice([cirq.NamedQubit(str(s)) for s in range(10)]).qid_pairs()) == 45


def test_qid_pair():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
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

    with pytest.raises(ValueError, match='A QidPair cannot have identical qids.'):
        cirq.SymmetricalQidPair(q0, q0)
