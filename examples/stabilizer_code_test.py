import cirq

import examples.stabilizer_code as sc


def encode_corrupt_correct(code, input_val, error_gate, error_loc):
    circuit = cirq.Circuit()
    qubits = [cirq.NamedQubit(str(i)) for i in range(code.n - code.k)] + [cirq.NamedQubit('c')]
    ancillas = [cirq.NamedQubit(f"d{i}") for i in range(code.n - code.k)]

    circuit += code.encode(qubits)

    if error_gate and error_loc:
        circuit.append(error_gate(qubits[error_loc]))

    circuit += code.correct(qubits, ancillas)

    results = cirq.Simulator().simulate(
        circuit, qubit_order=(qubits + ancillas), initial_state=(input_val * 2 ** len(ancillas))
    )

    decoded = code.decode(qubits, ancillas, results.state_vector())

    return decoded[0]


def test_no_error():
    # Table 3.2.
    five_qubit_code = sc.StabilizerCode(
        group_generators=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'], allowed_errors=['X', 'Z']
    )

    for input_val in [0, 1]:
        assert encode_corrupt_correct(five_qubit_code, input_val, None, None) == input_val


def test_errors():
    # Table 3.2.
    five_qubit_code = sc.StabilizerCode(
        group_generators=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'], allowed_errors=['X', 'Z']
    )

    for input_val in [0, 1]:
        for error_gate in [cirq.X, cirq.Z]:
            for error_loc in range(five_qubit_code.n):
                assert (
                    encode_corrupt_correct(five_qubit_code, input_val, error_gate, error_loc)
                    == input_val
                )


def test_imperfect_code():
    # Also known as the bit-flip code.
    bit_flip_code = sc.StabilizerCode(group_generators=['ZZI', 'ZIZ'], allowed_errors=['X'])

    for input_val in [0, 1]:
        for error_gate in [cirq.X, cirq.Z]:
            for error_loc in range(bit_flip_code.n):
                assert (
                    encode_corrupt_correct(bit_flip_code, input_val, error_gate, error_loc)
                    == input_val
                )
