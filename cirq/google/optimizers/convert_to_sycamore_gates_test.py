import numpy as np
import pytest

import cirq
import cirq.google.optimizers.convert_to_sycamore_gates as cgoc
import cirq.google.optimizers.textbook_gates_from_sycamore as cgot


def _assert_equivalent_op_tree(x: cirq.OP_TREE, y: cirq.OP_TREE):
    a = list(cirq.flatten_op_tree(x))
    b = list(cirq.flatten_op_tree(y))
    assert a == b


def test_known_two_q_operations_to_sycamore_operations_cz():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = cirq.CZ(qubits[0], qubits[1])
    test_op_tree = cirq.Circuit(operation)
    cgoc.ConvertToSycamoreGates().optimize_circuit(test_op_tree)
    true_op_tree = cirq.Circuit(cgot.decompose_cz_into_syc(
        qubits[0], qubits[1]))
    _assert_equivalent_op_tree(test_op_tree, true_op_tree)


def test_cz_unitary():
    q = cirq.LineQubit.range(2)
    circuit1 = cirq.Circuit(cirq.CZ(q[0], q[1]))
    circuit2 = circuit1.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(circuit2)
    assert cirq.allclose_up_to_global_phase(cirq.unitary(circuit1),
                                            cirq.unitary(circuit2))


def test_known_two_q_operations_to_sycamore_operations_cphase():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    for phi in np.linspace(0, 2 * np.pi, 30):
        operation = cirq.CZPowGate(exponent=phi / np.pi).on(
            qubits[0], qubits[1])
        test_op_tree = cirq.Circuit(operation)
        cgoc.ConvertToSycamoreGates().optimize_circuit(test_op_tree)
        true_op_tree = cirq.Circuit(cgot.cphase(phi, qubits[0], qubits[1]))
        _assert_equivalent_op_tree(test_op_tree, true_op_tree)


def test_cphase_unitary():
    q = cirq.LineQubit.range(2)
    for phi in np.linspace(0, 2 * np.pi, 30):
        operation = cirq.CZPowGate(exponent=phi / np.pi).on(q[0], q[1])
        circuit1 = cirq.Circuit(operation)
        circuit2 = circuit1.copy()
        cgoc.ConvertToSycamoreGates().optimize_circuit(circuit2)
        assert cirq.allclose_up_to_global_phase(cirq.unitary(circuit1),
                                                cirq.unitary(circuit2))


def test_known_two_q_operations_to_sycamore_operations_zz():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    for theta in np.linspace(0, 2 * np.pi, 30):
        operation = cirq.ZZPowGate(exponent=2 * theta / np.pi).on(
            qubits[0], qubits[1])
        test_op_tree = cirq.Circuit(operation)
        cgoc.ConvertToSycamoreGates().optimize_circuit(test_op_tree)
        true_op_tree = cirq.Circuit(cgot.rzz(theta, qubits[0], qubits[1]))
        _assert_equivalent_op_tree(test_op_tree, true_op_tree)


def test_zz_unitary():
    q = cirq.LineQubit.range(2)
    for theta in np.linspace(0, 2 * np.pi, 30):
        operation = cirq.ZZPowGate(exponent=2 * theta / np.pi).on(q[0], q[1])
        circuit1 = cirq.Circuit(operation)
        circuit2 = circuit1.copy()
        cgoc.ConvertToSycamoreGates().optimize_circuit(circuit2)
        assert cirq.allclose_up_to_global_phase(cirq.unitary(circuit1),
                                                cirq.unitary(circuit2))


def test_known_two_q_operations_to_sycamore_operations_swap():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = cirq.SWAP(qubits[0], qubits[1])
    test_op_tree = cirq.Circuit(operation)
    cgoc.ConvertToSycamoreGates().optimize_circuit(test_op_tree)
    true_op_tree = cirq.Circuit(
        cgot.decompose_swap_into_syc(qubits[0], qubits[1]))
    _assert_equivalent_op_tree(test_op_tree, true_op_tree)


def test_swap_unitary():
    q = cirq.LineQubit.range(2)
    circuit1 = cirq.Circuit(cirq.SWAP(q[0], q[1]))
    circuit2 = circuit1.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(circuit2)
    assert cirq.allclose_up_to_global_phase(cirq.unitary(circuit1),
                                            cirq.unitary(circuit2))


def test_known_two_q_operations_to_sycamore_operations_iswap():
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = cirq.ISWAP(qubits[0], qubits[1])
    test_op_tree = cirq.Circuit(operation)
    cgoc.ConvertToSycamoreGates().optimize_circuit(test_op_tree)
    true_op_tree = cirq.Circuit(
        cgot.decompose_iswap_into_syc(qubits[0], qubits[1]))
    _assert_equivalent_op_tree(test_op_tree, true_op_tree)


def test_iswap_unitary():
    q = cirq.LineQubit.range(2)
    circuit1 = cirq.Circuit(cirq.ISWAP(q[0], q[1]))
    circuit2 = circuit1.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(circuit2)
    assert cirq.allclose_up_to_global_phase(cirq.unitary(circuit1),
                                            cirq.unitary(circuit2))


def test_convert_to_sycamore_gates_swap_zz():
    qubits = cirq.LineQubit.range(3)

    gamma = np.random.randn()
    circuit1 = cirq.Circuit(cirq.SWAP(qubits[0], qubits[1]),
                            cirq.Z(qubits[2]),
                            cirq.ZZ(qubits[0], qubits[1])**gamma,
                            strategy=cirq.InsertStrategy.NEW)
    circuit2 = cirq.Circuit(cirq.ZZ(qubits[0], qubits[1])**gamma,
                            cirq.Z(qubits[2]),
                            cirq.SWAP(qubits[0], qubits[1]),
                            strategy=cirq.InsertStrategy.NEW)

    compiled_circuit1 = circuit1.copy()
    cgoc.ConvertToSycamoreGates()(compiled_circuit1)
    compiled_circuit2 = circuit2.copy()
    cgoc.ConvertToSycamoreGates()(compiled_circuit2)

    cirq.testing.assert_same_circuits(compiled_circuit1, compiled_circuit2)
    assert len(
        list(
            compiled_circuit1.findall_operations_with_gate_type(
                cirq.google.SycamoreGate))) == 3
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit1, compiled_circuit1, atol=1e-7)


def test_single_qubit_gate():
    q = cirq.LineQubit(0)
    mat = cirq.testing.random_unitary(2)
    gate = cirq.SingleQubitMatrixGate(mat)
    circuit = cirq.Circuit(gate(q))
    converted_circuit = circuit.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(converted_circuit)
    for op in converted_circuit.all_operations():
        gate = op.gate
        assert isinstance(gate, (cirq.PhasedXPowGate, cirq.ZPowGate))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, converted_circuit, atol=1e-8)


def test_unsupported_gate():

    class UnknownGate(cirq.TwoQubitGate):
        pass

    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    circuit = cirq.Circuit(UnknownGate()(q0, q1))
    with pytest.raises(ValueError, match='Unrecognized gate: '):
        cgoc.ConvertToSycamoreGates().optimize_circuit(circuit)


def test_non_gate_operation():

    class UnknownOperation(cirq.Operation):

        def __init__(self, qubits):
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            # coverage: ignore
            return UnknownOperation(self._qubits)

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(UnknownOperation([q0]))
    converted_circuit = circuit.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(converted_circuit)
    assert circuit == converted_circuit


def test_three_qubit_gate():

    class ThreeQubitGate(cirq.ThreeQubitGate):
        pass

    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)
    circuit = cirq.Circuit(ThreeQubitGate()(q0, q1, q2))
    with pytest.raises(TypeError):
        cgoc.ConvertToSycamoreGates().optimize_circuit(circuit)
