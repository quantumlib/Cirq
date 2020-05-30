import random
import cirq
"""Demonstrates a Hidden Shift algorithm.

Let f and g be two function oracles {0,1}^N -> {0,1}^N  which are the same
up to a hidden bit string s:

g(x) = f(x ⨁ s)

The Hidden Shift Algorithm determines s by quering the two oracles. The
implementation in this example considers the following oracle:

f(x) = Σ_i x_(2i-1) x_(2i)

where x_i is the i-th bit of x.

While a classical algorithm requires 2^(N/2) queries, the Hidden Shift
Algorithm solves the problem in O(1) steps. We thus have an exponential
reduction.

=== REFERENCE ===
[1] Wim van Dam, Sean Hallgreen, Lawrence Ip Quantum Algorithms for some
Hidden Shift Problems. https://arxiv.org/abs/quant-ph/0211140
[2] K Wrigth, et. a. Benchmarking an 11-qubit quantum computer.
Nature Communications, 107(28):12446–12450, 2010. doi:10.1038/s41467-019-13534-2


=== EXAMPLE OUTPUT ===
Secret shift sequence: [1, 0, 0, 1, 0, 1]
Circuit:
(0, 0): ───H───X───@───X───H───@───H───M('result')───
                   │           │       │
(1, 0): ───H───────@───────H───@───H───M─────────────
                                       │
(2, 0): ───H───────@───────H───@───H───M─────────────
                   │           │       │
(3, 0): ───H───X───@───X───H───@───H───M─────────────
                                       │
(4, 0): ───H───────@───────H───@───H───M─────────────
                   │           │       │
(5, 0): ───H───X───@───X───H───@───H───M─────────────
Sampled results:
Counter({'100101': 100})
Most common bitstring: 100101
Found a match: True
"""


def set_qubits(qubit_count):
    """Add the specified number of input qubits."""
    input_qubits = [cirq.GridQubit(i, 0) for i in range(qubit_count)]
    return input_qubits


def make_oracle_f(qubits):
    """Implement function {f(x) = Σ_i x_(2i-1) x_(2i)}."""
    return [
        cirq.CZ(qubits[2 * i], qubits[2 * i + 1])
        for i in range(len(qubits) // 2)
    ]


def make_hs_circuit(qubits, oracle_f, shift):
    """Find the shift between two almost equivalent functions."""
    c = cirq.Circuit()

    # Initialize qubits.
    c.append([
        cirq.H.on_each(*qubits),
    ])

    # Query oracle g: It is equivalent to that of f, shifted before and after:
    # Apply Shift:
    c.append(
        [cirq.X.on_each([qubits[k] for k in range(len(shift)) if shift[k]])])

    # Query oracle.
    c.append(oracle_f)

    # Apply Shift:
    c.append(
        [cirq.X.on_each([qubits[k] for k in range(len(shift)) if shift[k]])])

    # Second Application of Hadamards to apply inverse fct.
    c.append([
        cirq.H.on_each(*qubits),
    ])

    # Query oracle f (acting as inverse).
    c.append(oracle_f)

    # Final Application to go back to the single shift state:
    c.append([
        cirq.H.on_each(*qubits),
    ])

    # Measure the result.
    c.append(cirq.measure(*qubits, key='result'))

    return c


def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)


def main():
    qubit_count = 6
    sample_count = 100

    # Set up input qubits.
    input_qubits = set_qubits(qubit_count)

    # Define secret shift
    shift = [random.randint(0, 1) for _ in range(qubit_count)]
    print(f'Secret shift sequence: {shift}')

    # Make oracles (black box)
    oracle_f = make_oracle_f(input_qubits)

    # Embed the oracle into a quantum circuit implementing the Hidden Shift Algo
    circuit = make_hs_circuit(input_qubits, oracle_f, shift)
    print('Circuit:')
    print(circuit)

    # Sample from the circuit.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=sample_count)

    frequencies = result.histogram(key='result', fold_func=bitstring)
    print(f'Sampled results:\n{frequencies}')

    # Check if we actually found the secret value.
    most_common_bitstring = frequencies.most_common(1)[0][0]
    print(f'Most common bitstring: {most_common_bitstring}')
    print(f'Found a match: {most_common_bitstring == bitstring(shift)}')


if __name__ == '__main__':
    main()
