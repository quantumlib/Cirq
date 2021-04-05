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


def test_edges():
    class RawDevice(cirq.Device):
        pass

    assert RawDevice().edges is None

    class QubitFieldDevice(cirq.Device):
        def __init__(self, qubits):
            self.qubits = qubits

    assert len(QubitFieldDevice(cirq.LineQubit.range(10)).edges) == 9
    assert len(QubitFieldDevice(cirq.GridQubit.rect(10, 10)).edges) == 180
    assert len(QubitFieldDevice([cirq.NamedQubit(str(s)) for s in range(10)]).edges) == 45


def test_edge():
    e1 = cirq.QidPair(cirq.LineQubit(1), cirq.LineQubit(0))
    e2 = cirq.QidPair(cirq.LineQubit(0), cirq.LineQubit(1))
    e3 = cirq.QidPair(cirq.LineQubit(2), cirq.LineQubit(3))
    assert e1 == e2
    assert e2 != e3

    set1 = frozenset([e1, e2])
    set2 = frozenset([e2, e3])
    assert len(set1) == 1
    assert len(set2) == 2
