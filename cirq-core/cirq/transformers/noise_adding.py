from collections.abc import Mapping

import cirq
import numpy as np


def add_depolarizing_noise_to_two_qubit_gates(
    circuit: cirq.Circuit,
    p: float | Mapping[tuple[cirq.Qid, cirq.Qid], float],
    target_gate: cirq.Gate = cirq.CZ,
    rng: np.random.Generator | None = None,
) -> cirq.Circuit:
    """Add local depolarizing noise after two-qubit gates in a specified circuit. More specifically,
    with probability p, append a random non-identity two-qubit Pauli operator after each specified
    two-qubit gate.

    Args:
        circuit: The circuit to add noise to.
        p: The probability with which to add noise.
        target_gate: Add depolarizing nose after this type of gate
        rng: The pseudorandom number generator to use.

    Returns:
        The transformed circuit.
    """
    if rng is None:
        rng = np.random.default_rng()

    # add random Pauli gates with probability p after each of the specified gate
    assert target_gate.num_qubits() == 2, "`target_gate` must be a two-qubit gate."
    paulis = [cirq.I, cirq.X, cirq.Y, cirq.Z]
    new_moments = []
    for moment in circuit:
        new_moments.append(moment)
        if _gate_in_moment(target_gate, moment):
            # add a new moment with the Paulis
            target_pairs = {
                tuple(sorted(op.qubits)) for op in moment.operations if op.gate == target_gate
            }
            added_moment_ops = []
            for pair in target_pairs:
                if isinstance(p, float):
                    p_i = p
                elif isinstance(p, Mapping):
                    pair_sorted_tuple = (pair[0], pair[1])
                    p_i = p[pair_sorted_tuple]
                else:
                    raise TypeError(
                        "p must either be a float or a mapping from sorted qubit pairs to floats"
                    )
                apply = rng.choice([True, False], p=[p_i, 1 - p_i])
                if apply:
                    choices = [
                        (pauli_a(pair[0]), pauli_b(pair[1]))
                        for pauli_a in paulis
                        for pauli_b in paulis
                    ][1:]
                    pauli_to_apply = rng.choice(np.array(choices, dtype=object))
                    added_moment_ops.append(pauli_to_apply)
            if len(added_moment_ops) > 0:
                new_moments.append(cirq.Moment(*added_moment_ops))
    return cirq.Circuit.from_moments(*new_moments)
