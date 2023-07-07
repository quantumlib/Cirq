# pylint: disable=wrong-or-nonexistent-copyright-notice
# Used for classical post-processing:
from collections import Counter
import numpy as np
import scipy as sp
import cirq

"""Demonstrates Simon's algorithm.
Simon's Algorithm solves the following problem:

Given a function  f:{0,1}^n -> {0,1}^n, such that for some s ∈ {0,1}^n,

f(x) = f(y) iff  x ⨁ y ∈ {0^n, s},

find the n-bit string s.

A classical algorithm requires O(2^n/2) queries to find s, while Simon’s
algorithm needs only O(n) quantum queries.

=== REFERENCE ===
D. R. Simon. On the power of quantum cryptography. In35th FOCS, pages 116–123,
Santa Fe,New Mexico, 1994. IEEE Computer Society Press.

=== EXAMPLE OUTPUT ===
Secret string = [1, 0, 0, 1, 0, 0]
Circuit:
                ┌──────┐   ┌───────────┐
(0, 0): ────H────@──────────@─────@──────H───M('result')───
                 │          │     │          │
(1, 0): ────H────┼@─────────┼─────┼──────H───M─────────────
                 ││         │     │          │
(2, 0): ────H────┼┼@────────┼─────┼──────H───M─────────────
                 │││        │     │          │
(3, 0): ────H────┼┼┼@───────┼─────┼──────H───M─────────────
                 ││││       │     │          │
(4, 0): ────H────┼┼┼┼@──────┼─────┼──────H───M─────────────
                 │││││      │     │          │
(5, 0): ────H────┼┼┼┼┼@─────┼─────┼──────H───M─────────────
                 ││││││     │     │
(6, 0): ─────────X┼┼┼┼┼─────X─────┼───×────────────────────
                  │││││           │   │
(7, 0): ──────────X┼┼┼┼───────────┼───┼────────────────────
                   ││││           │   │
(8, 0): ───────────X┼┼┼───────────┼───┼────────────────────
                    │││           │   │
(9, 0): ────────────X┼┼───────────X───×────────────────────
                     ││
(10, 0): ────────────X┼────────────────────────────────────
                      │
(11, 0): ─────────────X────────────────────────────────────
                └──────┘   └───────────┘
Most common Simon Algorithm answer is: ('[1 0 0 1 0 0]', 100)

***If the input string is s=0^n, no significant answer can be
distinguished (since the null-space of the system of equations
provided by the measurements gives a random vector). This will
lead to low frequency count in output string.
"""


def main(qubit_count=3):

    data = []  # we'll store here the results

    # define a secret string:
    secret_string = np.random.randint(2, size=qubit_count)

    print(f'Secret string = {secret_string}')

    n_samples = 100
    for _ in range(n_samples):
        flag = False  # check if we have a linearly independent set of measures
        while not flag:
            # Choose qubits to use.
            input_qubits = [cirq.GridQubit(i, 0) for i in range(qubit_count)]  # input x
            output_qubits = [
                cirq.GridQubit(i + qubit_count, 0) for i in range(qubit_count)
            ]  # output f(x)

            # Pick coefficients for the oracle and create a circuit to query it.
            oracle = make_oracle(input_qubits, output_qubits, secret_string)

            # Embed oracle into special quantum circuit querying it exactly once
            circuit = make_simon_circuit(input_qubits, output_qubits, oracle)

            # Sample from the circuit a n-1 times (n = qubit_count).
            simulator = cirq.Simulator()
            results = [
                simulator.run(circuit).measurements['result'][0] for _ in range(qubit_count - 1)
            ]

            # Classical Post-Processing:
            flag = post_processing(data, results)

    freqs = Counter(data)
    print('Circuit:')
    print(circuit)
    print(f'Most common answer was : {freqs.most_common(1)[0]}')


def make_oracle(input_qubits, output_qubits, secret_string):
    """Gates implementing the function f(a) = f(b) iff a ⨁ b = s"""
    # Copy contents to output qubits:
    for control_qubit, target_qubit in zip(input_qubits, output_qubits):
        yield cirq.CNOT(control_qubit, target_qubit)

    # Create mapping:
    if sum(secret_string):  # check if the secret string is non-zero
        # Find significant bit of secret string (first non-zero bit)
        significant = list(secret_string).index(1)

        # Add secret string to input according to the significant bit:
        for j in range(len(secret_string)):
            if secret_string[j] > 0:
                yield cirq.CNOT(input_qubits[significant], output_qubits[j])
    # Apply a random permutation:
    pos = [
        0,
        len(secret_string) - 1,
    ]  # Swap some qubits to define oracle. We choose first and last:
    yield cirq.SWAP(output_qubits[pos[0]], output_qubits[pos[1]])


def make_simon_circuit(input_qubits, output_qubits, oracle):
    """Solves for the secret period s of a 2-to-1 function such that
    f(x) = f(y) iff x ⨁ y = s
    """

    c = cirq.Circuit()

    # Initialize qubits.
    c.append([cirq.H.on_each(*input_qubits)])

    # Query oracle.
    c.append(oracle)

    # Measure in X basis.
    c.append([cirq.H.on_each(*input_qubits), cirq.measure(*input_qubits, key='result')])

    return c


def post_processing(data, results):
    """Solves a system of equations with modulo 2 numbers"""
    sing_values = sp.linalg.svdvals(results)
    tolerance = 1e-5
    if sum(sing_values < tolerance) == 0:  # check if measurements are linearly dependent
        flag = True
        null_space = sp.linalg.null_space(results).T[0]
        solution = np.around(null_space, 3)  # chop very small values
        minval = abs(min(solution[np.nonzero(solution)], key=abs))
        solution = (solution / minval % 2).astype(int)  # renormalize vector mod 2
        data.append(str(solution))
        return flag


if __name__ == '__main__':
    main()
