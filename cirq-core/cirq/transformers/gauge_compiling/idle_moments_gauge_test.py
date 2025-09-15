import cirq
from cirq.transformers import gauge_compiling as gc


def test_add_gauge_merges_gates():
    tr = gc.IdleMomentsGauge(2, gauges='pauli')

    circuit = cirq.Circuit.from_moments([], [], [], cirq.X(cirq.q(0)), [], [], cirq.X(cirq.q(0)))
    transformed_circuit = tr(circuit, rng_or_seed=0)

    assert transformed_circuit == cirq.Circuit.from_moments(
        [],
        [],
        [],
        cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1, z_exponent=0)(cirq.q(0)),
        [],
        [],
        cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1, z_exponent=0)(cirq.q(0)),
    )


def test_add_gauge_respects_ignore_tag():
    tr = gc.IdleMomentsGauge(2, gauges='pauli')

    circuit = cirq.Circuit.from_moments(
        cirq.X(cirq.q(0)), [], [], cirq.X(cirq.q(0)).with_tags('ignore')
    )
    transformed_circuit = tr(
        circuit, context=cirq.TransformerContext(tags_to_ignore=("ignore",)), rng_or_seed=0
    )
    assert transformed_circuit == cirq.Circuit.from_moments(
        cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1, z_exponent=0)(cirq.q(0)),
        [],
        cirq.Z(cirq.q(0)),
        cirq.X(cirq.q(0)).with_tags('ignore'),
    )


def test_add_gauge_on_prefix():
    tr = gc.IdleMomentsGauge(3, gauges='clifford', gauge_beginning=True)

    circuit = cirq.Circuit.from_moments([], [], [], cirq.CNOT(cirq.q(0), cirq.q(1)))
    transformed_circuit = tr(circuit, rng_or_seed=0)
    assert transformed_circuit == cirq.Circuit.from_moments(
        [
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[20](cirq.q(0)),
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[15](cirq.q(1)),
        ],
        [],
        [
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[20](cirq.q(0)) ** -1,
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[15](cirq.q(1)) ** -1,
        ],
        cirq.CNOT(cirq.q(0), cirq.q(1)),
    )


def test_add_gauge_on_suffix():
    tr = gc.IdleMomentsGauge(3, gauges='inv_clifford', gauge_ending=True)

    circuit = cirq.Circuit.from_moments(cirq.CNOT(cirq.q(0), cirq.q(1)), [], [], [])
    transformed_circuit = tr(circuit, rng_or_seed=0)
    assert transformed_circuit == cirq.Circuit.from_moments(
        cirq.CNOT(cirq.q(0), cirq.q(1)),
        [
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[20](cirq.q(0)) ** -1,
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[15](cirq.q(1)) ** -1,
        ],
        [],
        [
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[20](cirq.q(0)),
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[15](cirq.q(1)),
        ],
    )


def test_add_gauge_respects_min_length():
    tr = gc.IdleMomentsGauge(2, gauges=[cirq.X])

    circuit = cirq.Circuit.from_moments(cirq.X(cirq.q(0)), [], cirq.X(cirq.q(0)))
    transformed_circuit = tr(circuit, rng_or_seed=0)
    assert transformed_circuit == circuit
