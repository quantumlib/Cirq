import pytest
import cirq


def test_valid_operations():
    adonis = cirq.iqm.Adonis()

    adonis.validate_operation(cirq.GateOperation(cirq.X,
                                                 [cirq.GridQubit(0, 1)]))

    adonis.validate_operation(
        cirq.GateOperation(
            cirq.CZ,
            [cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]))


def test_invalid_operations():
    adonis = cirq.iqm.Adonis()

    with pytest.raises(ValueError):
        adonis.validate_operation(
            cirq.GateOperation(cirq.H, [cirq.GridQubit(0, 1)]))

    with pytest.raises(ValueError):
        adonis.validate_operation(
            cirq.GateOperation(cirq.CNOT,
                               [cirq.GridQubit(1, 0),
                                cirq.GridQubit(1, 1)]))


def test_qubits_not_on_device():
    adonis = cirq.iqm.Adonis()

    with pytest.raises(ValueError):
        adonis.validate_operation(
            cirq.GateOperation(cirq.X, [cirq.GridQubit(0, 0)]))

    with pytest.raises(ValueError):
        adonis.validate_operation(
            cirq.GateOperation(cirq.CZ,
                               [cirq.GridQubit(2, 0),
                                cirq.GridQubit(2, 1)]))


def test_qubits_not_connected():
    adonis = cirq.iqm.Adonis()

    with pytest.raises(ValueError):
        adonis.validate_operation(
            cirq.GateOperation(cirq.CZ,
                               [cirq.GridQubit(0, 1),
                                cirq.GridQubit(1, 2)]))

    with pytest.raises(ValueError):
        adonis.validate_operation(
            cirq.GateOperation(cirq.ISWAP,
                               [cirq.GridQubit(1, 0),
                                cirq.GridQubit(1, 2)]))
