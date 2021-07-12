from typing import cast, List, Optional

import numpy as np
import pytest

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

    state_vector = (
        cirq.Simulator()
        .simulate(
            circuit, qubit_order=(qubits + ancillas), initial_state=(input_val * 2 ** len(ancillas))
        )
        .state_vector()
    )

    decoded = code.decode(qubits, ancillas, state_vector)

    # Trace out the syndrome out of the state.
    nq = len(qubits)
    na = len(ancillas)
    traced_out_state = np.sum(
        state_vector.reshape((2,) * (nq + na)), axis=tuple(range(nq, nq + na))
    ).reshape(-1)

    return decoded[0], traced_out_state


def test_no_error():
    # Table 3.2.
    five_qubit_code = sc.StabilizerCode(
        group_generators=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'], correctable_errors=['X', 'Z']
    )

    for input_val in [0, 1]:
        decoded, _ = encode_corrupt_correct(
            five_qubit_code, input_val, error_gate=None, error_loc=None
        )
        assert decoded == input_val


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
    code = sc.StabilizerCode(group_generators=group_generators, correctable_errors=['X', 'Z'])

    for input_val in [0, 1]:
        _, traced_out_state_no_error = encode_corrupt_correct(
            code, input_val, error_gate=None, error_loc=None
        )

        for error_gate in [cirq.X, cirq.Z]:
            for error_loc in range(code.n):
                decoded, traced_out_state = encode_corrupt_correct(
                    code, input_val, error_gate, error_loc
                )
                assert decoded == input_val
                np.testing.assert_allclose(traced_out_state_no_error, traced_out_state, atol=1e-6)


def test_imperfect_code():
    # Also known as the bit-flip code.
    bit_flip_code = sc.StabilizerCode(group_generators=['ZZI', 'ZIZ'], correctable_errors=['X'])

    for input_val in [0, 1]:
        _, traced_out_state_no_error = encode_corrupt_correct(
            bit_flip_code, input_val, error_gate=None, error_loc=None
        )

        for error_gate in [cirq.X]:
            for error_loc in range(bit_flip_code.n):
                decoded, traced_out_state = encode_corrupt_correct(
                    bit_flip_code, input_val, error_gate, error_loc
                )
                assert decoded == input_val
                np.testing.assert_allclose(traced_out_state_no_error, traced_out_state, atol=1e-6)

    # Test that we cannot correct a Z error. In this case, they manifest as a phase error, so we
    # test the state vectors.
    _, traced_out_state_no_error = encode_corrupt_correct(
        bit_flip_code, input_val=1, error_gate=None, error_loc=None
    )
    _, traced_out_state_z1_error = encode_corrupt_correct(
        bit_flip_code, input_val=1, error_gate=cirq.Z, error_loc=1
    )

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_allclose(traced_out_state_no_error, traced_out_state_z1_error, atol=1e-6)
