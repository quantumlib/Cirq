import random
import cirq

"""Example program that demonstrates a Hidden Shift algorithm.

The Hidden Shift Problem is one of the known problems whose quantum algorithm
solution shows exponential speedup over classical computing. Part of the
advantage lies on the ability to perform Fourier transforms efficiently. This
can be used to extract correlations between certain functions, as we will
demonstrate here:

Let f and g be two functions {0,1}^N -> {0,1}  which are the same
up to a hidden bit string s:

g(x) = f(x ⨁ s), for all x in {0,1}^N

The implementation in this example considers the following (so-called "bent")
functions:

f(x) = Σ_i x_(2i) x_(2i+1),

where x_i is the i-th bit of x and i runs from 0 to N/2 - 1.

While a classical algorithm requires 2^(N/2) queries, the Hidden Shift
Algorithm solves the problem in O(N) quantum operations. We describe below the
steps of the algorithm:

(1) Prepare the quantum state in the initial state |0⟩^N

(2) Make a superposition of all inputs |x⟩ with  a set of Hadamard gates, which
act as a (Quantum) Fourier Transform.

(3) Compute the shifted function g(x) = f(x ⨁ s) into the phase with a proper
set of gates. This is done first by shifting the state |x⟩ with X gates, then
implementing the bent function as a series of Controlled-Z gates, and finally
recovering the |x⟩ states with another set of X gates.

(4) Apply a Fourier Transform to generate another superposition of states with
an extra phase that is added to f(x ⨁ s).

(5) Query the oracle f into the phase with a proper set of controlled gates.
One can then prove that the phases simplify giving just a superposition with
a phase depending directly on the shift.

(6) Apply another set of Hadamard gates which act now as an Inverse Fourier
Transform to get the state |s⟩

(7) Measure the resulting state to get s.

Note that we only query g and f once to solve the problem.

=== REFERENCES ===
[1] Wim van Dam, Sean Hallgreen, Lawrence Ip Quantum Algorithms for some
Hidden Shift Problems. https://arxiv.org/abs/quant-ph/0211140
[2] K Wrigth, et. a. Benchmarking an 11-qubit quantum computer.
Nature Communications, 107(28):12446–12450, 2010. doi:10.1038/s41467-019-13534-2
[3] Rötteler, M. Quantum Algorithms for highly non-linear Boolean functions.
Proceedings of the 21st annual ACM-SIAM Symposium on Discrete Algorithms.
doi: 10.1137/1.9781611973075.37


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
    """Implement function {f(x) = Σ_i x_(2i) x_(2i+1)}."""
    return [cirq.CZ(qubits[2 * i], qubits[2 * i + 1]) for i in range(len(qubits) // 2)]


def make_hs_circuit(qubits, oracle_f, shift):
    """Find the shift between two almost equivalent functions."""
    c = cirq.Circuit()

    # Initialize qubits.
    c.append(
        [
            cirq.H.on_each(*qubits),
        ]
    )

    # Query oracle g: It is equivalent to that of f, shifted before and after:
    # Apply Shift:
    c.append([cirq.X.on_each([qubits[k] for k in range(len(shift)) if shift[k]])])

    # Query oracle.
    c.append(oracle_f)

    # Apply Shift:
    c.append([cirq.X.on_each([qubits[k] for k in range(len(shift)) if shift[k]])])

    # Second Application of Hadamards.
    c.append(
        [
            cirq.H.on_each(*qubits),
        ]
    )

    # Query oracle f (this simplifies the phase).
    c.append(oracle_f)

    # Inverse Fourier Transform with Hadamards to go back to the shift state:
    c.append(
        [
            cirq.H.on_each(*qubits),
        ]
    )

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

    # Embed oracle into quantum circuit implementing the Hidden Shift Algorithm
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
