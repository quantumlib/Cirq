"""Creates and simulates a simple circuit.

=== EXAMPLE OUTPUT ===
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
"""

import cirq


def main():
    # Pick a qubit.
    qubit = cirq.GridQubit(0, 0)

    # Create a circuit
    circuit = cirq.Circuit(
        cirq.X(qubit) ** 0.5, cirq.measure(qubit, key='m')  # Square root of NOT.  # Measurement.
    )
    print("Circuit:")
    print(circuit)

    # Simulate the circuit several times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=20)
    print("Results:")
    print(result)


if __name__ == '__main__':
    main()
