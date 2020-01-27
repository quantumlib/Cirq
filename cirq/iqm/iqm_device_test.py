import pytest
import cirq


class TestOperationValidation:
    adonis = cirq.iqm.Adonis()
    q0 = cirq.GridQubit(0, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(1, 1)
    q3 = cirq.GridQubit(1, 2)
    q4 = cirq.GridQubit(2, 1)

    def test_valid_operations(self):
        self.adonis.validate_operation(
            cirq.GateOperation(cirq.X, [self.q0]))

        self.adonis.validate_operation(
            cirq.GateOperation(cirq.YPowGate(exponent=0.25), [self.q0]))

        self.adonis.validate_operation(
            cirq.GateOperation(cirq.CZ, [self.q1, self.q2]))

        self.adonis.validate_operation(
            cirq.GateOperation(cirq.ISwapPowGate(exponent=0.5),
                               [self.q1, self.q2]))

    def test_invalid_operations(self):
        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.GateOperation(cirq.H, [self.q0]))

        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.GateOperation(cirq.CNOT, [self.q1, self.q2]))

    def test_qubits_not_on_device(self):
        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.GateOperation(cirq.X, [cirq.GridQubit(0, 0)]))

        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.GateOperation(cirq.CZ,
                                   [cirq.GridQubit(2, 0),
                                    cirq.GridQubit(2, 1)]))

    def test_qubits_not_connected(self):
        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.GateOperation(cirq.CZ, [self.q0, self.q3]))

        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.GateOperation(cirq.ISWAP, [self.q1, self.q3]))


class TestGateDecomposition:
    adonis = cirq.iqm.Adonis()
    q0 = cirq.GridQubit(0, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(1, 1)
    q3 = cirq.GridQubit(1, 2)
    q4 = cirq.GridQubit(2, 1)

    def test_native_single_qubit_gates(self):
        self.adonis.decompose_operation(
            cirq.XPowGate(exponent=0.75).on(self.q0))

    # TODO enable when this call halts
    # def test_unsupported_single_qubit_gates(self):
    #     self.adonis.decompose_operation(
    #         cirq.HPowGate(exponent=0.55).on(self.q0))

    def test_native_two_qubit_gate(self):
        self.adonis.decompose_operation(
            cirq.CZPowGate(exponent=0.15).on(self.q2, self.q4))

    # TODO enable when this call halts
    # def test_unsupported_two_qubit_gates(self):
    #     self.adonis.decompose_operation(
    #         cirq.CNOT.on(self.q2, self.q4))
