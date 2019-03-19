"""Quantum Teleportation.
Quantum Teleportation is a process by which a quantum state can be transmitted
by sending only two classical bits of information. This is accomplished by
pre-sharing an entangled state between the sender and the receiver. This
entangled state allows the receiver of the two classical bits of information
to possess a qubit with the same state as the one held by the sender.

In the following example output, qubit 0 is set to a random state by applying
X and Y gates. By sending two classical bits of information after qubit 0 and
qubit 1 are measured, the final state of qubit 2 will be identical to the
original random state of qubit 0. This is only possible given that an
entangled state is pre-shared between the sender and receiver.

=== REFERENCE ===
https://en.wikipedia.org/wiki/Quantum_teleportation
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.1895

=== EXAMPLE OUTPUT ===
Circuit:
0: ---X^0.25---Y^0.125-----------@---H---M-------@---
                                 |               |
1: ----------------------H---@---X-------M---@---|---
                             |               |   |
2: --------------------------X---------------X---@---

Bloch Sphere of Qubit 0 After Random X and Y Gates:
x:  0.2706 y:  -0.7071 z:  0.6533

Bloch Sphere of Qubit 2 at Final State:
x:  0.2706 y:  -0.7071 z:  0.6533

"""

import random
import numpy as np
import cirq


def make_quantum_teleportation_circuit():
    circuit = cirq.Circuit()
    q0, q1, q2 = cirq.LineQubit.range(3)

    # Creates a random state for q0
    circuit.append([cirq.X(q0)**random.random(), cirq.Y(q0)**random.random()])
    # Creates Bell State to be shared on q1 and q2
    circuit.append([cirq.H(q1), cirq.CNOT(q1, q2)])
    # Bell measurement of q0 (qubit to be teleported) and q1 (entangled qubit)
    circuit.append([cirq.CNOT(q0, q1), cirq.H(q0)])
    circuit.append([cirq.measure(q0), cirq.measure(q1)])
    # Uses q0 and q1 to perform operation on q2 to recover the state of q0
    circuit.append([cirq.CNOT(q1, q2), cirq.CZ(q0, q2)])

    return circuit


def main():
    state = []
    circuit = make_quantum_teleportation_circuit()

    print("Circuit:")
    print(circuit)

    sim = cirq.Simulator()

    # Records in a list each state of q0 for the simulation
    step_results = sim.simulate_moment_steps(circuit)
    for step in step_results:
        state.append(cirq.bloch_vector_from_state_vector(step.state_vector(), 0))

    print("\nBloch Sphere of Qubit 0 After Random X and Y Gates:")
    # Prints the Bloch Sphere of q0 after the X and Y gates
    b0X, b0Y, b0Z = state[1]
    print("x: ", np.around(b0X, 4),
          "y: ", np.around(b0Y, 4),
          "z: ", np.around(b0Z, 4))

    # Records the final state of the simulation
    final_results = sim.simulate(circuit)

    print("\nBloch Sphere of Qubit 2 at Final State:")
    # Prints the Bloch Sphere of q2 at the final state
    b2X, b2Y, b2Z = \
          cirq.bloch_vector_from_state_vector(final_results.final_state, 2)
    print("x: ", np.around(b2X, 4),
          "y: ", np.around(b2Y, 4),
          "z: ", np.around(b2Z, 4))


if __name__ == '__main__':
    main()
