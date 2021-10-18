import pytest
import numpy as np

import scipy.linalg

import cirq
import cirq_google
import cirq_google.optimizers.convert_to_sycamore_gates as cgoc
from cirq_google.optimizers.two_qubit_gates.gate_compilation import gate_product_tabulation

_rng = cirq.value.parse_random_state(11)  # for determinism


def test_convert_to_sycamore_gates_swap_zz():
    qubits = cirq.LineQubit.range(3)

    gamma = np.random.randn()
    circuit1 = cirq.Circuit(
        cirq.SWAP(qubits[0], qubits[1]),
        cirq.Z(qubits[2]),
        cirq.ZZ(qubits[0], qubits[1]) ** gamma,
        strategy=cirq.InsertStrategy.NEW,
    )
    circuit2 = cirq.Circuit(
        cirq.ZZ(qubits[0], qubits[1]) ** gamma,
        cirq.Z(qubits[2]),
        cirq.SWAP(qubits[0], qubits[1]),
        strategy=cirq.InsertStrategy.NEW,
    )

    compiled_circuit1 = circuit1.copy()
    cgoc.ConvertToSycamoreGates()(compiled_circuit1)
    compiled_circuit2 = circuit2.copy()
    cgoc.ConvertToSycamoreGates()(compiled_circuit2)

    cirq.testing.assert_same_circuits(compiled_circuit1, compiled_circuit2)
    assert (
        len(list(compiled_circuit1.findall_operations_with_gate_type(cirq_google.SycamoreGate)))
        == 3
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit1, compiled_circuit1, atol=1e-7
    )


def test_convert_to_sycamore_gates_fsim():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1))
    compiled_circuit = circuit.copy()
    cgoc.ConvertToSycamoreGates()(compiled_circuit)

    cirq.testing.assert_same_circuits(circuit, compiled_circuit)


def test_single_qubit_gate():
    q = cirq.LineQubit(0)
    mat = cirq.testing.random_unitary(2)
    gate = cirq.MatrixGate(mat, qid_shape=(2,))
    circuit = cirq.Circuit(gate(q))
    converted_circuit = circuit.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(converted_circuit)
    ops = list(converted_circuit.all_operations())
    assert len(ops) == 1
    assert isinstance(ops[0].gate, cirq.PhasedXZGate)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, converted_circuit, atol=1e-8
    )


def test_single_qubit_gate_phased_xz():
    q = cirq.LineQubit(0)
    gate = cirq.PhasedXZGate(axis_phase_exponent=0.2, x_exponent=0.3, z_exponent=0.4)
    circuit = cirq.Circuit(gate(q))
    converted_circuit = circuit.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(converted_circuit)
    ops = list(converted_circuit.all_operations())
    assert len(ops) == 1
    assert ops[0].gate == gate


def test_circuit_operation_inspection():
    q0, q1 = cirq.LineQubit.range(2)
    gate = cirq.PhasedXZGate(axis_phase_exponent=0.2, x_exponent=0.3, z_exponent=0.4)
    cop = cirq.CircuitOperation(cirq.FrozenCircuit(gate(q0)))
    assert cgoc.ConvertToSycamoreGates()._is_native_sycamore_op(cop)

    cop2 = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.SWAP(q0, q1)))
    assert not cgoc.ConvertToSycamoreGates()._is_native_sycamore_op(cop2)


def test_circuit_operation_conversion():
    q0, q1 = cirq.LineQubit.range(2)
    subcircuit = cirq.FrozenCircuit(cirq.X(q0), cirq.SWAP(q0, q1))
    circuit = cirq.Circuit(cirq.CircuitOperation(subcircuit))
    converted_circuit = circuit.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(converted_circuit)
    # Verify that the CircuitOperation was preserved.
    ops = list(converted_circuit.all_operations())
    assert isinstance(ops[0], cirq.CircuitOperation)
    # Verify that the contents of the CircuitOperation were optimized.
    reconverted_subcircuit = ops[0].circuit.unfreeze().copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(reconverted_subcircuit)
    assert ops[0].circuit == reconverted_subcircuit
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, converted_circuit, atol=1e-8
    )


def test_unsupported_gate():
    class UnknownGate(cirq.testing.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(UnknownGate()(q0, q1))
    with pytest.raises(ValueError, match='Unrecognized gate: '):
        cgoc.ConvertToSycamoreGates().optimize_circuit(circuit)


def test_nested_unsupported_gate():
    class UnknownGate(cirq.testing.TwoQubitGate):
        pass

    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    subcircuit = cirq.FrozenCircuit(UnknownGate()(q0, q1))
    circuit = cirq.Circuit(cirq.CircuitOperation(subcircuit))
    with pytest.raises(ValueError, match='Unrecognized gate: '):
        cgoc.ConvertToSycamoreGates().optimize_circuit(circuit)


def test_unsupported_phased_iswap():
    """Tests that a Phased ISwap with a provided phase_exponent and exponent is
    not supported."""
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    circuit = cirq.Circuit(cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=0.33)(q0, q1))
    with pytest.raises(ValueError, match='phase_exponent of .25 OR an exponent of 1'):
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
    class ThreeQubitGate(cirq.testing.ThreeQubitGate):
        pass

    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)
    circuit = cirq.Circuit(ThreeQubitGate()(q0, q1, q2))
    with pytest.raises(TypeError):
        cgoc.ConvertToSycamoreGates().optimize_circuit(circuit)


def random_single_qubit_unitary():
    a, b, c = np.random.random(3) * 2 * np.pi
    circuit = cirq.unitary(cirq.rz(a)) @ cirq.unitary(cirq.ry(b)) @ cirq.unitary(cirq.rz(c))
    assert np.allclose(circuit.conj().T @ circuit, np.eye(2))
    return circuit


def test_unitary_decomp():
    q = cirq.GridQubit(0, 0)
    random_unitary = random_single_qubit_unitary()
    circuit = cirq.Circuit(
        [term.on(q) for term in cirq.single_qubit_matrix_to_gates(random_unitary)]
    )
    assert np.isclose(abs(np.trace(cirq.unitary(circuit).conj().T @ random_unitary)), 2.0)


def test_zztheta():
    zz = np.kron(cirq.unitary(cirq.Z), cirq.unitary(cirq.Z))
    qubits = cirq.LineQubit.range(2)
    for theta in np.linspace(0, 2 * np.pi, 10):
        expected_unitary = scipy.linalg.expm(-1j * theta * zz)
        circuit = cirq.Circuit(cgoc.rzz(theta, qubits[0], qubits[1]))
        actual_unitary = cirq.unitary(circuit)
        cirq.testing.assert_allclose_up_to_global_phase(actual_unitary, expected_unitary, atol=1e-7)


def test_zztheta_zzpow():
    qubits = cirq.LineQubit.range(2)
    for theta in np.linspace(0, 2 * np.pi, 10):
        syc_circuit = cirq.Circuit(cgoc.rzz(theta, qubits[0], qubits[1]))
        cirq_circuit = cirq.Circuit(
            [cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5).on(*qubits)]
        )
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.unitary(cirq_circuit), cirq.unitary(syc_circuit), atol=1e-7
        )


def test_zztheta_qaoa_like():
    qubits = cirq.LineQubit.range(4)
    for exponent in np.linspace(-1, 1, 10):
        cirq_circuit = cirq.Circuit(
            [
                cirq.H.on_each(qubits),
                cirq.ZZPowGate(exponent=exponent)(qubits[0], qubits[1]),
                cirq.ZZPowGate(exponent=exponent)(qubits[2], qubits[3]),
                cirq.rx(0.123).on_each(qubits),
            ]
        )
        syc_circuit = cirq_circuit.copy()
        cgoc.ConvertToSycamoreGates().optimize_circuit(syc_circuit)

        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.unitary(cirq_circuit), cirq.unitary(syc_circuit), atol=1e-7
        )


def test_zztheta_zzpow_unsorted_qubits():

    qubits = cirq.LineQubit(1), cirq.LineQubit(0)
    exponent = 0.06366197723675814
    expected_circuit = cirq.Circuit(
        cirq.ZZPowGate(exponent=exponent, global_shift=-0.5).on(qubits[0], qubits[1]),
    )
    actual_circuit = expected_circuit.copy()
    cgoc.ConvertToSycamoreGates().optimize_circuit(actual_circuit)

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(expected_circuit), cirq.unitary(actual_circuit), atol=1e-7
    )


def test_swap_zztheta():
    """Construct a Ising gate followed by a swap using a sycamore."""
    qubits = cirq.LineQubit.range(2)
    a, b = qubits
    for theta in np.linspace(0, 2 * np.pi, 10):
        expected_circuit = cirq.Circuit(
            cirq.SWAP(a, b), cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5).on(a, b)
        )
        expected_unitary = cirq.unitary(expected_circuit)
        actual_circuit = expected_circuit.copy()
        cgoc.ConvertToSycamoreGates().optimize_circuit(actual_circuit)
        actual_unitary = cirq.unitary(actual_circuit)
        cirq.testing.assert_allclose_up_to_global_phase(actual_unitary, expected_unitary, atol=1e-7)


def test_known_two_q_operations_to_sycamore_operations_cnot():
    a, b = cirq.LineQubit.range(2)
    op = cirq.CNOT(a, b)
    decomposed = cirq.Circuit(cgoc.ConvertToSycamoreGates().convert(op))

    # Should be equivalent.
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(op), cirq.unitary(decomposed), atol=1e-8
    )

    # Should have decomposed into two Sycamores.
    multi_qubit_ops = [e for e in decomposed.all_operations() if len(e.qubits) > 1]
    assert len(multi_qubit_ops) == 2
    assert all(isinstance(e.gate, cirq_google.SycamoreGate) for e in multi_qubit_ops)


@pytest.mark.parametrize(
    'gate',
    [
        cirq.MatrixGate(cirq.unitary(cirq.CX), qid_shape=(2, 2)),
        cirq.ISWAP,
        cirq.SWAP,
        cirq.CNOT,
        cirq.CZ,
        cirq.PhasedISwapPowGate(exponent=1.0),
        cirq.PhasedISwapPowGate(exponent=1.0, phase_exponent=0.33),
        cirq.PhasedISwapPowGate(exponent=0.66, phase_exponent=0.25),
        *[cirq.givens(theta) for theta in np.linspace(0, 2 * np.pi, 30)],
        *[cirq.ZZPowGate(exponent=2 * phi / np.pi) for phi in np.linspace(0, 2 * np.pi, 30)],
        *[cirq.CZPowGate(exponent=phi / np.pi) for phi in np.linspace(0, 2 * np.pi, 30)],
    ],
)
def test_convert_to_sycamore_equivalent_unitaries(gate):
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = gate.on(qubits[0], qubits[1])
    converted = cgoc.ConvertToSycamoreGates().convert(operation)
    u1 = cirq.unitary(cirq.Circuit(converted))
    u2 = cirq.unitary(operation)
    cirq.testing.assert_allclose_up_to_global_phase(u1, u2, atol=1e-8)


def test_convert_to_sycamore_tabulation():
    # A tabulation for the sycamore gate with an infidelity of .1.
    sycamore_tabulation = gate_product_tabulation(
        cirq.unitary(cirq_google.SYC), 0.1, random_state=_rng
    )
    qubits = [cirq.NamedQubit('a'), cirq.NamedQubit('b')]
    operation = cirq.MatrixGate(cirq.unitary(cirq.CX), qid_shape=(2, 2)).on(qubits[0], qubits[1])
    converted = cgoc.ConvertToSycamoreGates(sycamore_tabulation).convert(operation)
    u1 = cirq.unitary(cirq.Circuit(converted))
    u2 = cirq.unitary(operation)
    overlap = abs(np.trace(u1.conj().T @ u2))
    assert np.isclose(overlap, 4.0, 0.1)


def test_sycamore_invalid_tabulation():
    # An object other than a tabulation.
    sycamore_tabulation = {}
    with pytest.raises(ValueError):
        cgoc.ConvertToSycamoreGates(sycamore_tabulation)


q = cirq.GridQubit.rect(1, 3)
matrix_gate = cirq.MatrixGate(cirq.testing.random_unitary(2))


@pytest.mark.parametrize(
    'op, is_valid',
    [
        (cirq.CircuitOperation(cirq.FrozenCircuit(matrix_gate(q[0]))), True),
        (matrix_gate(q[0]), True),
        (matrix_gate(q[0]).with_tags('test_tags'), True),
        (matrix_gate(q[0]).controlled_by(q[1]), True),
        (matrix_gate(q[0]).controlled_by(q[1]).with_tags('test_tags'), True),
        (matrix_gate(q[0]).with_tags('test_tags').controlled_by(q[1]), True),
    ],
)
def test_supported_operation(op, is_valid):
    c = cirq.Circuit(op)
    assert (cirq_google.ConvertToSycamoreGates().optimization_at(c, 0, op) is not None) == is_valid
