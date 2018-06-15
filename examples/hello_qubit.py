"""Creates and simulates a simple circuit.

=== EXAMPLE OUTPUT ===
Circuit:
(0, 0): ───X^0.5───M───
Results:
m=11000111111011001000
"""

import cirq


def main():
    # Pick a qubit.
    qubit = cirq.google.XmonQubit(0, 0)

    # Create a circuit
    circuit = cirq.Circuit.from_ops(
        cirq.X(qubit)**0.5,  # Square root of NOT.
        cirq.MeasurementGate('m').on(qubit)  # Measurement.
    )
    print("Circuit:")
    print(circuit)

    # Simulate the circuit several times.
    simulator = cirq.google.XmonSimulator()
    result = simulator.run(circuit, repetitions=20)
    print("Results:")
    print(result)


if __name__ == '__main__':
    main()
