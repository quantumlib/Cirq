import pytest
import cirq


class TestOperationValidation:
    def test_valid_operations(self):
        adonis = cirq.iqm.Adonis()

        adonis.validate_operation(cirq.GateOperation(cirq.X,
                                                     [cirq.GridQubit(0, 1)]))

        adonis.validate_operation(
            cirq.GateOperation(cirq.YPowGate(exponent=0.25),
                               [cirq.GridQubit(0, 1)]))

        adonis.validate_operation(
            cirq.GateOperation(
                cirq.CZ,
                [cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]))

        adonis.validate_operation(
            cirq.GateOperation(
                cirq.ISwapPowGate(exponent=0.5),
                [cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]))

    def test_invalid_operations(self):
        adonis = cirq.iqm.Adonis()

        with pytest.raises(ValueError):
            adonis.validate_operation(
                cirq.GateOperation(cirq.H, [cirq.GridQubit(0, 1)]))

        with pytest.raises(ValueError):
            adonis.validate_operation(
                cirq.GateOperation(cirq.CNOT,
                                   [cirq.GridQubit(1, 0),
                                    cirq.GridQubit(1, 1)]))

    def test_qubits_not_on_device(self):
        adonis = cirq.iqm.Adonis()

        with pytest.raises(ValueError):
            adonis.validate_operation(
                cirq.GateOperation(cirq.X, [cirq.GridQubit(0, 0)]))

        with pytest.raises(ValueError):
            adonis.validate_operation(
                cirq.GateOperation(cirq.CZ,
                                   [cirq.GridQubit(2, 0),
                                    cirq.GridQubit(2, 1)]))

    def test_qubits_not_connected(self):
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


class TestGateDecomposition:
    adonis = cirq.iqm.Adonis()
    q0 = cirq.GridQubit(0, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(1, 1)
    q3 = cirq.GridQubit(1, 2)
    q4 = cirq.GridQubit(2, 1)

    def test_native_single_qubit_gates(self):
        self.adonis.decompose_operation(cirq.XPowGate(exponent=0.75).on(self.q0))

    # TODO enable when this call halts
    # def test_unsupported_single_qubit_gates(self):
    #     self.adonis.decompose_operation(cirq.HPowGate(exponent=0.55).on(self.q0))
