import cirq

import examples.stabilizer_code as sc

def encode_corrupt_correct(input_val, error_gate, error_loc):
    code = sc.StabilitizerCode(group_generators=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'])

    circuit = cirq.Circuit()
    qubits = [cirq.NamedQubit(name) for name in ['0', '1', '2', '3', 'c']]
    ancillas = [cirq.NamedQubit(name) for name in ['d0', 'd1', 'd2', 'd3']]

    code.encode(circuit, qubits)
    if error_gate and error_loc:
        circuit.append(error_gate(qubits[error_loc]))

    code.correct(circuit, qubits, ancillas)

    results = cirq.Simulator().simulate(
        circuit, qubit_order=(qubits + ancillas), initial_state=(input_val * 2 ** 4)
    )

    qubit_map = {qubit: i for i, qubit in enumerate(qubits + ancillas)}
    pauli_string = cirq.PauliString(dict(zip(qubits, code.Z[0])))
    trace = pauli_string.expectation_from_state_vector(results.state_vector(), qubit_map)

    print(cirq.dirac_notation(4*results.state_vector()))
    print(trace)

    return round((1 - trace.real) / 2)

def test_no_error():
    for input_val in [0, 1]:
        assert encode_corrupt_correct(input_val, None, None) == input_val

def test_simple_errors():
    for input_val in [0, 1]:
        for error_gate in [cirq.X, cirq.Z]:
            for error_loc in range(5):
                assert encode_corrupt_correct(input_val, error_gate, error_loc) == input_val
