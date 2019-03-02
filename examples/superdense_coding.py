"""Superdense Coding.
Superdense Coding is a method to transmit two classical bits of information
from a sender to a receiver by sending only one qubit. This is accomplished
by pre-sharing an entangled qubit.

The follwoing example sets qubit 0 to 0 and qubit 1 to 1 and are output in
qubits 3 and 4.

=== REFERENCE ===
https://en.m.wikipedia.org/wiki/Superdense_coding

=== EXAMPLE OUTPUT ===
0: ───────────M───────@───────────────────
                      │
1: ───X───────M───@───┼───────────────────
                  │   │
2: ───H───@───────X───@───×───────────────
          │               │
3: ───────┼───────────────×───@───H───M───
          │                   │
4: ───────X───────────────────X───────M───

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

    circuit.append([cirq.X(q1), cirq.H(q2), cirq.CNOT(q2, q4)])
    circuit.append([cirq.measure(q0), cirq.measure(q1)])
    circuit.append([cirq.CNOT(q1, q2)])
    circuit.append([cirq.CZ(q0, q2)])
    circuit.append([cirq.SWAP(q2, q3)])
    circuit.append([cirq.CNOT(q3, q4)])
    circuit.append([cirq.H(q3)])
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
