import pytest
import cirq
from cirq import iqm


class TestOperationValidation:
    adonis = iqm.Adonis()
    q0 = cirq.GridQubit(0, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(1, 1)
    q3 = cirq.GridQubit(1, 2)
    q4 = cirq.GridQubit(2, 1)

    def test_valid_operations(self):
        self.adonis.validate_operation(cirq.X(self.q0))
        self.adonis.validate_operation(cirq.Y(self.q2))
        self.adonis.validate_operation(cirq.Z(self.q4))

        self.adonis.validate_operation(cirq.YPowGate(exponent=0.25)(self.q0))

        self.adonis.validate_operation(cirq.CZ(self.q1, self.q2))

        self.adonis.validate_operation(
            cirq.CZPowGate(exponent=0.5)(self.q1, self.q2))

        self.adonis.validate_operation(cirq.measure(self.q0))
        self.adonis.validate_operation(cirq.measure(self.q1, key='test'))

    def test_invalid_operations(self):
        with pytest.raises(ValueError):
            self.adonis.validate_operation(cirq.H(self.q0))

        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.PauliString([cirq.X(self.q0), cirq.Y(self.q0)]))

        with pytest.raises(ValueError):
            self.adonis.validate_operation(cirq.CNOT(self.q1, self.q2))

    def test_qubits_not_on_device(self):
        with pytest.raises(ValueError):
            self.adonis.validate_operation(cirq.X(cirq.GridQubit(0, 0)))

        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.CZ(cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)))

    def test_qubits_not_connected(self):
        with pytest.raises(ValueError):
            self.adonis.validate_operation(cirq.CZ(self.q0, self.q3))

        with pytest.raises(ValueError):
            self.adonis.validate_operation(
                cirq.CZPowGate(exponent=0.11)(self.q1, self.q3))


class TestGateDecomposition:
    adonis = iqm.Adonis()
    q0 = cirq.GridQubit(0, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(1, 1)
    q3 = cirq.GridQubit(1, 2)
    q4 = cirq.GridQubit(2, 1)

    @staticmethod
    def is_native(op_or_op_list) -> bool:
        if iqm.Adonis.is_native_operation(op_or_op_list):
            return True
        for op in op_or_op_list:
            if not iqm.Adonis.is_native_operation(op):
                return False  # coverage: ignore
        return True

    def test_native_single_qubit_gates(self):
        decomposition_xpow = self.adonis.decompose_operation(
            cirq.XPowGate(exponent=0.75).on(self.q0))
        assert TestGateDecomposition.is_native(decomposition_xpow)

        decomposition_z = self.adonis.decompose_operation(cirq.Z.on(self.q0))
        assert TestGateDecomposition.is_native(decomposition_z)

    def test_unsupported_single_qubit_gates(self):
        decomposition_hpow = self.adonis.decompose_operation(
            cirq.HPowGate(exponent=-0.55).on(self.q1))
        assert TestGateDecomposition.is_native(decomposition_hpow)

        decomposition_phasedxz = self.adonis.decompose_operation(
            cirq.PhasedXZGate(x_exponent=0.2,
                              z_exponent=-0.5,
                              axis_phase_exponent=0.75).on(self.q1))
        assert TestGateDecomposition.is_native(decomposition_phasedxz)

    def test_native_two_qubit_gate(self):
        decomposition = self.adonis.decompose_operation(
            cirq.CZPowGate(exponent=0.15).on(self.q2, self.q4))
        assert TestGateDecomposition.is_native(decomposition)

    def test_unsupported_two_qubit_gates(self):
        decomposition_cnot = self.adonis.decompose_operation(
            cirq.CNOT.on(self.q1, self.q2))
        assert TestGateDecomposition.is_native(decomposition_cnot)

        decomposition_cx = self.adonis.decompose_operation(
            cirq.SWAP.on(self.q3, self.q2))
        assert TestGateDecomposition.is_native(decomposition_cx)


class TestCircuitValidation:
    adonis = iqm.Adonis()
    q0 = cirq.GridQubit(0, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(1, 1)

    def test_valid_circuit(self):
        valid_circuit = cirq.Circuit(device=self.adonis)
        valid_circuit.append(cirq.Y(self.q0))
        valid_circuit.append(cirq.measure(self.q0, key='a'))
        valid_circuit.append(cirq.measure(self.q1, key='b'))

        self.adonis.validate_circuit(valid_circuit)

    def test_invalid_circuit(self):
        invalid_circuit = cirq.Circuit(device=self.adonis)
        invalid_circuit.append(cirq.Y(self.q0))
        invalid_circuit.append(cirq.measure(self.q0, key='a'))
        invalid_circuit.append(cirq.measure(self.q1, key='a'))

        with pytest.raises(ValueError):
            self.adonis.validate_circuit(invalid_circuit)
