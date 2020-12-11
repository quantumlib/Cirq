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
import numpy as np
import cirq


def make_quantum_teleportation_circuit(ranX, ranY):
    circuit = cirq.Circuit()
    msg, alice, bob = cirq.LineQubit.range(3)

    # Creates Bell state to be shared between Alice and Bob.
    circuit.append([cirq.H(alice), cirq.CNOT(alice, bob)])
    # Creates a random state for the Message.
    circuit.append([cirq.X(msg) ** ranX, cirq.Y(msg) ** ranY])
    # Bell measurement of the Message and Alice's entangled qubit.
    circuit.append([cirq.CNOT(msg, alice), cirq.H(msg)])
    circuit.append(cirq.measure(msg, alice))
    # Uses the two classical bits from the Bell measurement to recover the
    # original quantum Message on Bob's entangled qubit.
    circuit.append([cirq.CNOT(alice, bob), cirq.CZ(msg, bob)])

    return circuit


def main(seed=None):
    """Run a simple simulation of quantum teleportation.

    Args:
        seed: The seed to use for the simulation.
    """
    random.seed(seed)

    ranX = random.random()
    ranY = random.random()
    circuit = make_quantum_teleportation_circuit(ranX, ranY)

    print("Circuit:")
    print(circuit)

    sim = cirq.Simulator(seed=seed)

    # Run a simple simulation that applies the random X and Y gates that
    # create our message.
    q0 = cirq.LineQubit(0)
    message = sim.simulate(cirq.Circuit([cirq.X(q0) ** ranX, cirq.Y(q0) ** ranY]))

    print("\nBloch Sphere of Message After Random X and Y Gates:")
    # Prints the Bloch Sphere of the Message after the X and Y gates.
    expected = cirq.bloch_vector_from_state_vector(message.final_state_vector, 0)
    print(
        "x: ",
        np.around(expected[0], 4),
        "y: ",
        np.around(expected[1], 4),
        "z: ",
        np.around(expected[2], 4),
    )

    # Records the final state of the simulation.
    final_results = sim.simulate(circuit)

    print("\nBloch Sphere of Qubit 2 at Final State:")
    # Prints the Bloch Sphere of Bob's entangled qubit at the final state.
    teleported = cirq.bloch_vector_from_state_vector(final_results.final_state_vector, 2)
    print(
        "x: ",
        np.around(teleported[0], 4),
        "y: ",
        np.around(teleported[1], 4),
        "z: ",
        np.around(teleported[2], 4),
    )

    return expected, teleported


if __name__ == '__main__':
    main()
