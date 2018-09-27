from cirq import GridQubit
from cirq import ops
import cirq.experiments.generate_google_v2_random_circuit as random_circuit_v2


def test_generate_google_v2_random_circuit():
    circuit = random_circuit_v2.generate_random_v2_circuit_grid(
        n_rows=4, n_cols=5, cz_depth=9, seed=0)
    # We check that is exactly circuit inst_4x5_10_0
    # in github.com/sboixo/GRCS cz_v2
    assert len(circuit) == 11
    assert len(list(circuit.findall_operations_with_gate_type(
        ops.Rot11Gate))) == 35
    assert len(list(circuit.findall_operations_with_gate_type(
        ops.RotXGate))) == 15
    assert len(list(circuit.findall_operations_with_gate_type(
        ops.RotYGate))) == 23
    assert len(list(circuit.findall_operations_with_gate_type(
        ops.RotZGate))) == 32
    assert len(list(circuit.findall_operations_with_gate_type(
        ops.HGate))) == 40
    qubits = [GridQubit(i, j) for i in range(4)
              for j in range(5)]
    assert circuit.operation_at(qubits[0], 1).gate == ops.CZ
    assert circuit.operation_at(qubits[5], 2).gate == ops.CZ
    assert circuit.operation_at(qubits[8], 3).gate == ops.CZ
    assert circuit.operation_at(qubits[13], 4).gate == ops.CZ
    assert circuit.operation_at(qubits[12], 5).gate == ops.CZ
    assert circuit.operation_at(qubits[13], 6).gate == ops.CZ
    assert circuit.operation_at(qubits[14], 7).gate == ops.CZ
