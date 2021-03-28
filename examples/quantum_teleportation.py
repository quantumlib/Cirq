"""Quantum Teleportation.
Quantum Teleportation is a process by which a quantum state can be transmitted
by sending only two classical bits of information. This is accomplished by
pre-sharing an entangled state between the sender (Alice) and the receiver
(Bob). This entangled state allows the receiver (Bob) of the two classical
bits of information to possess a qubit with the same state as the one held by
the sender (Alice).

In the following example output, qubit 0 (the Message) is set to a random state
by applying X and Y gates. By sending two classical bits of information after
qubit 0 (the Message) and qubit 1 (Alice's entangled qubit) are measured, the
final state of qubit 2 (Bob's entangled qubit) will be identical to the
original random state of qubit 0 (the Message). This is only possible given
that an entangled state is pre-shared between Alice and Bob.

=== REFERENCE ===
https://en.wikipedia.org/wiki/Quantum_teleportation
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.1895

=== EXAMPLE OUTPUT ===
Circuit:
0: -----------X^0.25---Y^0.125---@---H---M-------@---
                                 |       |       |
1: ---H---@----------------------X-------M---@---|---
          |                                  |   |
2: -------X----------------------------------X---@---

Bloch Sphere of Message After Random X and Y Gates:
x:  0.2706 y:  -0.7071 z:  0.6533

Bloch Sphere of Qubit 2 at Final State:
x:  0.2706 y:  -0.7071 z:  0.6533

"""

import random
from typing import Tuple

import numpy as np
import cirq
from cirq.sim.simulator import (
    TStepResult,
    SimulatesIntermediateState,
    TSimulationTrialResult,
    TSimulatorState,
    TActOnArgs,
)


def _print_bloch(vector):
    print(
        'x: ',
        np.around(vector[0], 4),
        'y: ',
        np.around(vector[1], 4),
        'z: ',
        np.around(vector[2], 4),
    )


def _run(
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ]
) -> Tuple[TSimulationTrialResult, TStepResult]:
    # Initialize our qubit state space.
    msg, alice, bob = qubits = cirq.LineQubit.range(3)
    qubit_order = cirq.QubitOrder.as_qubit_order(qubits)

    # First we create a bell state circuit and simulate it on the qubits.
    bell_circuit = cirq.Circuit(cirq.H(alice), cirq.CNOT(alice, bob))
    args = sim._create_act_on_args(bell_circuit, 0, qubit_order)
    list(sim._core_iterator(bell_circuit, args, qubit_order))
    print('\nBell Circuit:')
    print(bell_circuit)

    # Second we randomize the message qubit.
    rand_x = random.random()
    rand_y = random.random()
    msg_circuit = cirq.Circuit(
        cirq.X(msg) ** rand_x,
        cirq.Y(msg) ** rand_y,
    )
    list(sim._core_iterator(msg_circuit, args, qubit_order))
    print('\nMessage Circuit:')
    print(msg_circuit)

    # Now we measure on Alice's side
    alice_circuit = cirq.Circuit(
        cirq.CNOT(msg, alice),
        cirq.H(msg),
        cirq.measure(alice, key='alice'),
        cirq.measure(msg, key='msg'),
    )
    alice_results = list(sim._core_iterator(alice_circuit, args, qubit_order))
    meas_alice = alice_results[1].measurements['alice'] == [1]
    meas_msg = alice_results[2].measurements['msg'] == [1]
    print('\nAlice Circuit:')
    print(alice_circuit)
    print(f'meas_alice={meas_alice}')
    print(f'meas_msg={meas_msg}')

    # Finally we construct Bob's circuit based on Alice's measurements
    bob_circuit = cirq.Circuit()
    if meas_alice:
        bob_circuit.append(cirq.X(bob))  # coverage: ignore

    if meas_msg:
        bob_circuit.append(cirq.Z(bob))  # coverage: ignore

    *_, final_results = sim._core_iterator(bob_circuit, args, qubit_order)
    print('\nBob Circuit:')
    print(bob_circuit)

    # We simulate our message circuit separately for comparison
    message = sim.simulate(msg_circuit)

    return message, final_results


def main(seed=None):
    """Run a simple simulation of quantum teleportation.

    Args:
        seed: The seed to use for the simulation.
    """
    random.seed(seed)

    # Run with density matrix simulator
    print('***Run with density matrix simulator***')
    sim = cirq.DensityMatrixSimulator(seed=seed)
    message, final_results = _run(sim)
    print('\nBloch Sphere of Message After Random X and Y Gates:')
    expected = cirq.bloch_vector_from_state_vector(message.final_density_matrix, 0)
    _print_bloch(expected)
    print('\nBloch Sphere of Qubit 2 at Final State:')
    teleported = cirq.bloch_vector_from_state_vector(final_results._density_matrix, 2)
    _print_bloch(teleported)

    # Run with sparse simulator
    print('\n\n\n\n\n***Run with sparse simulator***')
    sim = cirq.Simulator(seed=seed)
    message, final_results = _run(sim)
    print('\nBloch Sphere of Message After Random X and Y Gates:')
    expected = cirq.bloch_vector_from_state_vector(message.final_state_vector, 0)
    _print_bloch(expected)
    print('\nBloch Sphere of Qubit 2 at Final State:')
    teleported = cirq.bloch_vector_from_state_vector(final_results.state_vector(), 2)
    _print_bloch(teleported)

    return expected, teleported


if __name__ == '__main__':
    main()
