# pylint: disable=wrong-or-nonexistent-copyright-notice

"""Demonstrates the Bernstein-Vazirani algorithm.

The (non-recursive) Bernstein-Vazirani algorithm takes a black-box oracle
implementing a function f(a) = a·factors + bias (mod 2), where 'bias' is 0 or 1,
'a' and 'factors' are vectors with all elements equal to 0 or 1, and the
algorithm solves for 'factors' in a single query to the oracle.

=== REFERENCE ===

Bernstein, Ethan, and Umesh Vazirani. "Quantum complexity theory."
SIAM Journal on Computing 26.5 (1997): 1411-1473.

=== EXAMPLE OUTPUT ===

Secret function:
f(a) = a·<0, 1, 1, 1, 0, 0, 1, 0> + 1 (mod 2)
Circuit:
(0, 0): ───────H───────────────────────H───M───
                                           │
(1, 0): ───────H───────@───────────────H───M───
                       │                   │
(2, 0): ───────H───────┼───@───────────H───M───
                       │   │               │
(3, 0): ───────H───────┼───┼───@───────H───M───
                       │   │   │           │
(4, 0): ───────H───────┼───┼───┼───────H───M───
                       │   │   │           │
(5, 0): ───────H───────┼───┼───┼───────H───M───
                       │   │   │           │
(6, 0): ───────H───────┼───┼───┼───@───H───M───
                       │   │   │   │       │
(7, 0): ───────H───────┼───┼───┼───┼───H───M───
                       │   │   │   │
(8, 0): ───X───H───X───X───X───X───X───────────
Sampled results:
Counter({'01110010': 3})
Most common matches secret factors:
True
"""

import random

import cirq


def main(qubit_count=8):
    circuit_sample_count = 3

    # Choose qubits to use.
    input_qubits = [cirq.GridQubit(i, 0) for i in range(qubit_count)]
    output_qubit = cirq.GridQubit(qubit_count, 0)

    # Pick coefficients for the oracle and create a circuit to query it.
    secret_bias_bit = random.randint(0, 1)
    secret_factor_bits = [random.randint(0, 1) for _ in range(qubit_count)]
    oracle = make_oracle(input_qubits, output_qubit, secret_factor_bits, secret_bias_bit)
    print(
        'Secret function:\nf(a) = '
        f"a·<{', '.join(str(e) for e in secret_factor_bits)}> + "
        f"{secret_bias_bit} (mod 2)"
    )

    # Embed the oracle into a special quantum circuit querying it exactly once.
    circuit = make_bernstein_vazirani_circuit(input_qubits, output_qubit, oracle)
    print('Circuit:')
    print(circuit)

    # Sample from the circuit a couple times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=circuit_sample_count)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    print(f'Sampled results:\n{frequencies}')

    # Check if we actually found the secret value.
    most_common_bitstring = frequencies.most_common(1)[0][0]
    print(
        'Most common matches secret factors:\n'
        f'{most_common_bitstring == bitstring(secret_factor_bits)}'
    )


def make_oracle(input_qubits, output_qubit, secret_factor_bits, secret_bias_bit):
    """Gates implementing the function f(a) = a·factors + bias (mod 2)."""

    if secret_bias_bit:
        yield cirq.X(output_qubit)

    for qubit, bit in zip(input_qubits, secret_factor_bits):
        if bit:  # pragma: no cover
            yield cirq.CNOT(qubit, output_qubit)


def make_bernstein_vazirani_circuit(input_qubits, output_qubit, oracle):
    """Solves for factors in f(a) = a·factors + bias (mod 2) with one query."""

    c = cirq.Circuit()

    # Initialize qubits.
    c.append([cirq.X(output_qubit), cirq.H(output_qubit), cirq.H.on_each(*input_qubits)])

    # Query oracle.
    c.append(oracle)

    # Measure in X basis.
    c.append([cirq.H.on_each(*input_qubits), cirq.measure(*input_qubits, key='result')])

    return c


def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)


if __name__ == '__main__':
    main()
