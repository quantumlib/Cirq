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


def test_deprecation():
    class QubitFieldDevice(cirq.Device):
        def __init__(self):
            self.qubits = cirq.LineQubit.range(1)

    with cirq.testing.assert_deprecated("qubits", "deprecated", deadline="v0.12"):
        _ = QubitFieldDevice().qubit_set()

    class PrivateQubitFieldDevice(cirq.Device):
        def __init__(self):
            self._qubits = cirq.LineQubit.range(1)

    with cirq.testing.assert_deprecated("_qubits", "deprecated", deadline="v0.12"):
        _ = PrivateQubitFieldDevice().qubit_set()
