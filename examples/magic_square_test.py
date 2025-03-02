# pylint: disable=wrong-or-nonexistent-copyright-notice
import cirq
import numpy as np

import examples.magic_square as ms


def run_magic_square_game() -> None:
    """Test that Alice and Bob win 100% of the time with a noiseless simulator."""

    sampler = cirq.Simulator()
    alice_qubits = cirq.GridQubit.rect(1, 4, 0, 0)  # test with 2 measure qubits
    bob_qubits = cirq.GridQubit.rect(1, 4, 1, 0)
    for add_dd in [False, True]:
        result = ms.run_magic_square_game(sampler, alice_qubits, bob_qubits, add_dd=add_dd)
        assert np.all(result.get_win_matrix() == np.ones((3, 3)))
