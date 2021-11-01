# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Superdense Coding.
Superdense Coding is a method to transmit two classical bits of information
by sending only one qubit of information. This is accomplished by
pre-sharing an entangled state between the sender and the receiver. This
entangled state allows the receiver of the one qubit of information to
decode the two classical bits that were originally encoded by the sender.

In the following example output, qubit 0 and qubit 1 are randomly set to
either 0 or 1 by using Hadamard and Measure gates. By sending one qubit of
information between qubit 2 and qubit 3, the receiver is able to decode the
originally encoded information when qubits 3 and 4 are measured. This is
only possible given that an entangled state is pre-shared between the
sender and receiver.

The two input bits and the two output bits returned by this circuit should
be identical.

=== REFERENCE ===
https://en.m.wikipedia.org/wiki/Superdense_coding

=== EXAMPLE OUTPUT ===
Circuit:
0: ---H---M('input ')--------------@-----------------------------
          |                        |
1: ---H---M--------------------@---|-----------------------------
                               |   |
2: --------------------H---@---X---@---x-------------------------
                           |           |
3: ------------------------|-----------x---@---H---M('output')---
                           |               |       |
4: ------------------------X---------------X-------M-------------

Results:
input =10001000000000011110, 10000001100000001000
output=10001000000000011110, 10000001100000001000

"""

import cirq


def make_superdense_circuit():
    circuit = cirq.Circuit()
    q0, q1, q2, q3, q4 = cirq.LineQubit.range(5)

    # Randomly sets q0 and q1 to either 0 or 1
    circuit.append([cirq.H(q0), cirq.H(q1)])
    circuit.append(cirq.measure(q0, q1, key="input "))

    # Creates Bell State to be shared on q2 and q4
    circuit.append([cirq.H(q2), cirq.CNOT(q2, q4)])
    # Step 1 of encoding (controlled NOT gate on q1 / q2)
    circuit.append(cirq.CNOT(q1, q2))
    # Step 2 of encoding (controlled Z gate on q0 / q2)
    circuit.append(cirq.CZ(q0, q2))
    # Sends encoded information to receiver
    circuit.append(cirq.SWAP(q2, q3))
    # Step 1 of decoding (controlled NOT gate on q3 and q4)
    circuit.append(cirq.CNOT(q3, q4))
    # Step 2 of decoding (Hadamard gate on q3)
    circuit.append(cirq.H(q3))
    # Measurement by receiver to decode bits
    circuit.append(cirq.measure(q3, q4, key="output"))

    return circuit


def main():
    circuit = make_superdense_circuit()
    print("Circuit:")
    print(circuit)

    sim = cirq.Simulator()
    results = sim.run(circuit, repetitions=20)
    print("\nResults:")
    print(results)


if __name__ == '__main__':
    main()
