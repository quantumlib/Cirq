"""Demonstrates Grover algorithm.

The Grover algorithm takes a black-box oracle implementing a function 
{f(x) = 1 if x==x', f(x) = 0 if x!= x'} and finds x' within a randomly
ordered sequence of N items using O(sqrt(N)) operations and O(N log(N)) gates,
with the probability p >= 2/3.

At the moment, only 2-bit sequences (for which one pass through Grover operator
is enough) are considered.

=== REFERENCE ===
Coles, Eidenbenz et al. Quantum Algorithm Implementations for Beginners
https://arxiv.org/pdf/1804.03719.pdf

=== EXAMPLE OUTPUT ===
Secret bit sequence: [0, 0]
Circuit:
(0, 0): ───────H───Z───@───H───X───────@───────X───H───M───
                                             │               │               │
(1, 0): ───────H───Z───@───H───X───H───X───H───X───H───M───
                                             │
(2, 0): ───X───H───────X───────────────────────────────────
Sampled results:
Counter({'00': 10})
Most common bitstring: 00
Found a match: True

"""

import random
import cirq

def set_io_qubits(qubit_count):
    """Add the specified number of input and output qubits."""
    input_qubits = [cirq.google.XmonQubit(i, 0) for i in range(qubit_count)]
    output_qubit = cirq.google.XmonQubit(qubit_count, 0)
    return (input_qubits, output_qubit)

def make_oracle(input_qubits, output_qubit, x_bits):
    """Implement function {f(x) = 1 if x==x', f(x) = 0 if x!= x'}."""
    # Make oracle.
    # for (1, 1) it's just a Toffoli gate
    # otherwise negate the zero-bits.
    yield(cirq.Z(q) for (q, bit) in zip(input_qubits, x_bits) if not bit)
    yield(cirq.TOFFOLI(input_qubits[0], input_qubits[1], output_qubit))


def make_grover_circuit(input_qubits, output_qubit, oracle):
    """Find the value recognized by the oracle in sqrt(N) attempts."""
    # For 2 input qubits, that means using Grover operator only once.
    c = cirq.Circuit()

    # Initialize qubits.
    c.append([
        cirq.X(output_qubit),
        cirq.H(output_qubit),
        cirq.H.on_each(input_qubits),
    ])

    # Query oracle.
    c.append(oracle)

    # Construct Grover operator.
    c.append(cirq.H.on_each(input_qubits))
    c.append(cirq.X.on_each(input_qubits))
    c.append(cirq.H.on(input_qubits[1]))
    c.append(cirq.CNOT(input_qubits[0], input_qubits[1]))
    c.append(cirq.H.on(input_qubits[1]))
    c.append(cirq.X.on_each(input_qubits))
    c.append(cirq.H.on_each(input_qubits))

    # Measure the result.
    c.append(cirq.measure(*input_qubits, key='result'))

    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

def main():
    qubit_count = 2
    circuit_sample_count = 10

    #Set up input and output qubits.
    (input_qubits, output_qubit) = set_io_qubits(qubit_count)

    #Choose the x' and make an oracle which can recognize it.
    x_bits = [random.randint(0, 1) for _ in range(qubit_count)]
    print('Secret bit sequence: {}'.format(x_bits))

    # Make oracle (black box)
    oracle = make_oracle(input_qubits, output_qubit, x_bits)

    # Embed the oracle into a quantum circuit implementing Grover's algorithm.
    circuit = make_grover_circuit(input_qubits, output_qubit, oracle)
    print('Circuit:')
    print(circuit)

    # Sample from the circuit a couple times.
    simulator = cirq.google.XmonSimulator()
    result = simulator.run(circuit, repetitions=circuit_sample_count)

    frequencies = result.histogram(key='result', fold_func=bitstring)
    print('Sampled results:\n{}'.format(frequencies))

    # Check if we actually found the secret value.
    most_common_bitstring = frequencies.most_common(1)[0][0]
    print('Most common bitstring: {}'.format(most_common_bitstring))
    print('Found a match: {}'.format(most_common_bitstring == bitstring(x_bits)))

if __name__ == '__main__':
    main()
