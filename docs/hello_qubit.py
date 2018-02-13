import cirq


def main():

    # Define a qubit.
    qubit = cirq.QubitLoc(0, 0)

    # Create a circuit (qubits start in the |0> state).
    circuit = cirq.Circuit()
    circuit.append([
        # Square root of NOT.
        cirq.ExpWGate(half_turns=0.5).on(qubit),

        # Measurement.
        cirq.MeasurementGate('result').on(qubit)
    ])
    print("Circuit:")
    print(cirq.to_ascii(circuit))

    # Now simulate the circuit and print out the measurement result.
    simulator = cirq.sim.google.xmon_simulator.Simulator()
    results = []
    for _ in range(100):
        result = simulator.run(circuit).measurements['result'][0]
        results.append('1' if result else '0')
    print("Simulated measurement results:")
    print(''.join(results))


if __name__ == "__main__":
    main()
