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
        self.adonis.validate_operation(cirq.GateOperation(cirq.X, [self.q0]))

        self.adonis.validate_operation(
            cirq.GateOperation(cirq.YPowGate(exponent=0.25), [self.q0]))

        self.adonis.validate_operation(
            cirq.GateOperation(cirq.CZ, [self.q1, self.q2]))

        self.adonis.validate_operation(
            cirq.GateOperation(cirq.CZPowGate(exponent=0.5),
                               [self.q1, self.q2]))

    def test_invalid_operations(self):
        with pytest.raises(ValueError):
            self.adonis.validate_operation(cirq.GateOperation(
                cirq.H, [self.q0]))

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
                cirq.GateOperation(cirq.CZPowGate(exponent=0.11),
                                   [self.q1, self.q3]))


class TestGateDecomposition:
    adonis = iqm.Adonis()
    q0 = cirq.GridQubit(0, 1)
    q1 = cirq.GridQubit(1, 0)
    q2 = cirq.GridQubit(1, 1)
    q3 = cirq.GridQubit(1, 2)
    q4 = cirq.GridQubit(2, 1)

    def test_native_single_qubit_gates(self):
        decomposition_xpow = self.adonis.decompose_operation(
            cirq.XPowGate(exponent=0.75).on(self.q0))
        if iqm.Adonis.is_native_operation(decomposition_xpow):
            return
        for gate in decomposition_xpow:
            assert iqm.Adonis.is_native_operation(gate)

        decomposition_z = self.adonis.decompose_operation(cirq.Z.on(self.q0))
        if iqm.Adonis.is_native_operation(decomposition_z):
            return
        for gate in decomposition_z:
            assert iqm.Adonis.is_native_operation(gate)

    def test_unsupported_single_qubit_gates(self):
        decomposition_hpow = self.adonis.decompose_operation(
            cirq.HPowGate(exponent=-0.55).on(self.q1))
        if iqm.Adonis.is_native_operation(decomposition_hpow):
            return
        for gate in decomposition_hpow:
            assert iqm.Adonis.is_native_operation(gate)

        decomposition_phasedxz = self.adonis.decompose_operation(
            cirq.PhasedXZGate(x_exponent=0.2,
                              z_exponent=-0.5,
                              axis_phase_exponent=0.75).on(self.q1))
        if iqm.Adonis.is_native_operation(decomposition_phasedxz):
            return
        for gate in decomposition_phasedxz:
            assert iqm.Adonis.is_native_operation(gate)

    def test_native_two_qubit_gate(self):
        decomposition = self.adonis.decompose_operation(
            cirq.CZPowGate(exponent=0.15).on(self.q2, self.q4))
        if iqm.Adonis.is_native_operation(decomposition):
            return
        for gate in decomposition:
            assert iqm.Adonis.is_native_operation(gate)

    def test_unsupported_two_qubit_gates(self):
        decomposition_cnot = self.adonis.decompose_operation(
            cirq.CNOT.on(self.q1, self.q2))
        if iqm.Adonis.is_native_operation(decomposition_cnot):
            return
        for gate in decomposition_cnot:
            assert iqm.Adonis.is_native_operation(gate)

        decomposition_cx = self.adonis.decompose_operation(
            cirq.SWAP.on(self.q3, self.q2))
        if iqm.Adonis.is_native_operation(decomposition_cx):
            return
        for gate in decomposition_cx:
            assert iqm.Adonis.is_native_operation(gate)
