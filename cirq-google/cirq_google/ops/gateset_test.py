import cirq
import cirq_google


def test_basic():
    gs = cirq_google.Gateset(
        'mock_device',
        cirq.CZ,
        cirq_google.SYC,
        cirq.PhasedXZGate,
    )
    g = cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0)
    assert g in gs

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(g(q))
    assert gs.validate(circuit)

    assert gs.num_qubits == 2

    assert gs.default_target_gateset is not None
    assert len(gs.target_gatesets) == 2

    q0, q1 = cirq.LineQubit(0), cirq.LineQubit(1)
    circuit = cirq.Circuit(cirq.ISWAP(q0, q1))
    cirq.optimize_for_target_gateset(circuit, gateset=gs)
    cirq.optimize_for_target_gateset(circuit, gateset=gs.target_gatesets[1])
