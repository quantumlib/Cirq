import cirq.circuits as circuits
import cirq.ops as ops
import cirq.transformers as transformers


def _create_odd_ghz(qubits: list[ops.Qid]) -> circuits.Circuit:
    """Circuit to create a GHZ state on an odd number of qubits with 1D connectivity. Example:


    0: ───────────────────────────────H───@───H───
                                          │
    1: ───────────────────────H───@───H───@───────
                                  │
    2: ───────────────H───@───H───@───────────────
                          │
    3: ───H───────@───H───@───────────────────────
                  │
    4: ───H───@───@───────────────────────────────
              │
    5: ───H───@───────H───@───────────────────────
                          │
    6: ───────────────H───@───H───@───────────────
                                  │
    7: ───────────────────────H───@───H───@───────
                                          │
    8: ───────────────────────────────H───@───H───

    Args:
        qubits: A list of qubits such that CZ gates are possible between qubits[i] and qubits[i+1].

    Returns:
        A circuit to prepare the GHZ state.
    """

    nq = len(qubits)
    assert nq % 2 == 1 and nq >= 3
    center_idx = nq // 2
    moments = [
        circuits.Moment(
            ops.H(qubits[center_idx]), ops.H(qubits[center_idx + 1]), ops.H(qubits[center_idx - 1])
        ),
        circuits.Moment(ops.CZ(qubits[center_idx], qubits[center_idx + 1])),
        circuits.Moment(ops.CZ(qubits[center_idx], qubits[center_idx - 1])),
    ]
    for d in range(2, nq // 2 + 1):
        operations = [
            ops.H(qubits[center_idx + d - 1]),
            ops.H(qubits[center_idx + d]),
            ops.H(qubits[center_idx - d + 1]),
            ops.H(qubits[center_idx - d]),
        ]
        moments.append(circuits.Moment(*operations))
        moments.append(
            circuits.Moment(
                ops.CZ(qubits[center_idx + d - 1], qubits[center_idx + d]),
                ops.CZ(qubits[center_idx - d + 1], qubits[center_idx - d]),
            )
        )

    operations = [ops.H(qubits[0]), ops.H(qubits[-1])]
    moments.append(circuits.Moment(*operations))

    return circuits.Circuit.from_moments(*moments)


def _create_even_ghz(qubits: list[ops.Qid]) -> circuits.Circuit:
    """Circuit to create a GHZ state on an even number of qubits with 1D connectivity. Example:


    0: ───────────────────────────H───@───H───
                                      │
    1: ───────────────────H───@───H───@───────
                              │
    2: ───────────H───@───H───@───────────────
                      │
    3: ───H───@───────@───────────────────────
              │
    4: ───H───@───H───@───────────────────────
                      │
    5: ───────────H───@───H───@───────────────
                              │
    6: ───────────────────H───@───H───@───────
                                      │
    7: ───────────────────────────H───@───H───

    Args:
        qubits: A list of qubits such that CZ gates are possible between qubits[i] and qubits[i+1].

    Returns:
        A circuit to prepare the GHZ state.
    """

    nq = len(qubits)
    assert nq % 2 == 0 and nq >= 2
    center_idx = nq // 2
    moments = [
        circuits.Moment(ops.H(qubits[center_idx - 1]), ops.H(qubits[center_idx])),
        circuits.Moment(ops.CZ(qubits[center_idx - 1], qubits[center_idx])),
    ]
    for d in range(1, nq // 2):
        if d == 1:
            moments.append(
                circuits.Moment(
                    ops.H.on_each(
                        qubits[center_idx], qubits[center_idx + 1], qubits[center_idx - 2]
                    )
                )
            )
        else:
            operations = ops.H.on_each(
                qubits[center_idx - d - 1],
                qubits[center_idx - d],
                qubits[center_idx + d],
                qubits[center_idx + d - 1],
            )
            moments.append(circuits.Moment(operations))
        moments.append(
            circuits.Moment(
                ops.CZ(qubits[center_idx - d - 1], qubits[center_idx - d]),
                ops.CZ(qubits[center_idx + d], qubits[center_idx + d - 1]),
            )
        )

    operations = [ops.H(qubits[0]), ops.H(qubits[-1])] if nq > 2 else [ops.H(qubits[0])]
    moments.append(circuits.Moment(*operations))

    return circuits.Circuit.from_moments(*moments)


def _create_ghz_from_one_end(qubits: list[ops.Qid]) -> circuits.Circuit:
    """Circuit to create a GHZ state from one end in a 1D chain. Example:


    0: ───H───@───────────────────────────────────────────────────────
              │
    1: ───H───@───H───@───────────────────────────────────────────────
                      │
    2: ───────────H───@───H───@───────────────────────────────────────
                              │
    3: ───────────────────H───@───H───@───────────────────────────────
                                      │
    4: ───────────────────────────H───@───H───@───────────────────────
                                              │
    5: ───────────────────────────────────H───@───H───@───────────────
                                                      │
    6: ───────────────────────────────────────────H───@───H───@───────
                                                              │
    7: ───────────────────────────────────────────────────H───@───H───


    Args:
        qubits: A list of qubits such that CZ gates are possible between qubits[i] and qubits[i+1].

    Returns:
        A circuit to prepare the GHZ state.

    Raises:
        NotImplementedError: If requesting x_basis_cheat=True and x_basis
    """

    num_qubits = len(qubits)
    moments = []
    for cycle in range(num_qubits - 1):
        moments.append(circuits.Moment(ops.H.on_each(qubits[cycle : cycle + 2])))
        moments.append(circuits.Moment(ops.CZ(*qubits[cycle : (cycle + 2)])))
    moments.append(circuits.Moment(ops.H(qubits[-1])))

    return circuits.Circuit.from_moments(*moments)


def generate_1d_ghz_circuit(
    qubits: list[ops.Qid],
    add_dd: bool = True,
    dd_sequence: tuple[ops.Gate, ...] = (ops.X, ops.Y, ops.X, ops.Y),
    from_one_end: bool = False,
) -> circuits.Circuit:
    """Circuit to create a GHZ state on qubits with 1D connectivity.

    Args:
        qubits: A list of qubits such that CZ gates are possible between qubits[i] and qubits[i+1].
        add_dd: Whether to add dynamical decoupling to the circuit, done by adding gates.
        dd_sequence: The sequence of gates to insert for dynamical decoupling.
        from_one_end: Whether to grow the GHZ state from one end instead of the center.

    Returns:
        A circuit to prepare the GHZ state.
    """
    if from_one_end:
        circuit = _create_ghz_from_one_end(qubits)
    elif len(qubits) % 2 == 0:
        circuit = _create_even_ghz(qubits)
    else:
        circuit = _create_odd_ghz(qubits)

    if add_dd:
        # first add cirq.I in final moment to help with transformer:
        circuit[-1] += ops.I.on_each(set(qubits) - circuit[-1].qubits)

        # next, add dd
        circuit = transformers.dynamical_decoupling.add_dynamical_decoupling(
            circuit, schema=dd_sequence, single_qubit_gate_moments_only=True
        )

    return circuit
