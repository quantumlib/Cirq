"""Superdense Coding.
Superdense Coding is a method to transmit two classical bits of information
from a sender to a receiver by sending only one qubit. This is accomplished
by pre-sharing an entangled qubit.

The following example sets qubit 0 to 0 and qubit 1 to 1 and are output in
qubits 3 and 4.

=== REFERENCE ===
https://en.m.wikipedia.org/wiki/Superdense_coding

=== EXAMPLE OUTPUT ===
Circuit:
0: ---------------M-------@-------------------
                          |
1: ---X-----------M---@---|-------------------
                      |   |
2: -------H---@-------X---@---x---------------
              |               |
3: -----------|---------------x---@---H---M---
              |                   |
4: -----------X-------------------X-------M---

Results:
0=0
1=1
3=0
4=1

"""

import cirq


def make_superdense_circuit():
    circuit = cirq.Circuit()
    (q0, q1, q2, q3, q4) = cirq.LineQubit.range(5)

    # Sets q1 to 1 (and leaves q0 at 0)
    circuit.append([cirq.X(q1)])
    # Creates Bell State to be shared on q2 and q4
    circuit.append([cirq.H(q2), cirq.CNOT(q2, q4)])
    # Measures q0 and q1, both of which will be sent to the receiver
    circuit.append([cirq.measure(q0), cirq.measure(q1)])
    # Step 1 of encoding (controlled NOT gate on q1 / q2)
    circuit.append([cirq.CNOT(q1, q2)])
    # Step 2 of encoding (controlled Z gate on q0 / q2)
    circuit.append([cirq.CZ(q0, q2)])
    # Sends encoded information to receiver
    circuit.append([cirq.SWAP(q2, q3)])
    # Step 1 of decoding (controlled NOT gate on q3 and q4)
    circuit.append([cirq.CNOT(q3, q4)])
    # Step 2 of decoding (Hadamard gate on q3)
    circuit.append([cirq.H(q3)])
    # Measurement by receiver to decode bits
    circuit.append([cirq.measure(q3), cirq.measure(q4)])

    return circuit


def main():

    circuit = make_superdense_circuit()
    print("Circuit:")
    print(circuit)

    sim = cirq.Simulator()
    results = sim.run(circuit)
    print("\nResults:")
    print(results)


if __name__ == '__main__':
    main()
