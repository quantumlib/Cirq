import pytest
from typing import cast, List, Optional

import cirq

import examples.stabilizer_code as sc


def encode_corrupt_correct(
    code: sc.StabilizerCode,
    input_val: int,
    error_gate: Optional[cirq.SingleQubitCliffordGate],
    error_loc: int,
):
    circuit = cirq.Circuit()
    additional_qubits: List[cirq.Qid] = cast(
        List[cirq.Qid],
        [cirq.NamedQubit(str(i)) for i in range(code.n - code.k)],
    )
    unencoded_qubits: List[cirq.Qid] = cast(
        List[cirq.Qid],
        [cirq.NamedQubit('c')],
    )
    qubits = additional_qubits + unencoded_qubits
    ancillas: List[cirq.Qid] = cast(
        List[cirq.Qid], [cirq.NamedQubit(f"d{i}") for i in range(code.n - code.k)]
    )

    circuit += code.encode(additional_qubits, unencoded_qubits)

    if error_gate:
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


@pytest.mark.parametrize(
    'group_generators',
    [
        # Five qubit code, table 3.2 in thesis.
        (['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']),
        # Steane code.
        (['XXXXIII', 'XXIIXXI', 'XIXIXIX', 'ZZZZIII', 'ZZIIZZI', 'ZIZIZIZ']),
    ],
)
def test_errors(group_generators):
    code = sc.StabilizerCode(group_generators=group_generators, allowed_errors=['X', 'Z'])

    for input_val in [0, 1]:
        for error_gate in [cirq.X, cirq.Z]:
            for error_loc in range(code.n):
                assert encode_corrupt_correct(code, input_val, error_gate, error_loc) == input_val


def test_imperfect_code():
    # Also known as the bit-flip code.
    bit_flip_code = sc.StabilizerCode(group_generators=['ZZI', 'ZIZ'], allowed_errors=['X'])

    for input_val in [0, 1]:
        for error_gate in [cirq.X]:
            for error_loc in range(bit_flip_code.n):
                assert (
                    encode_corrupt_correct(bit_flip_code, input_val, error_gate, error_loc)
                    == input_val
                )

    # NOTE(tonybruguier): Even though the state vector shows that we cannot correct the error,
    # the decoded qubit is incorrect but only because it has the wrong phase. Since the Pauli
    # string measurement doesn't capture the phase it doesn't detect the error.
    # TODO(tonybruguier): Have a unit test that captures the fact that this code cannot correct
    # phase errors.
