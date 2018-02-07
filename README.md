# Cirq

Cirq is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

## Installation

Follow these [instructions](docs/install.md).

## Hello Qubit

A simple example to get you up and running:
```python
import cirq

# Define a qubit.
qubit = cirq.ops.QubitLoc(0, 0)

# Create a circuit (qubits start in the |0> state).
circuit = cirq.circuits.Circuit()
circuit.append([
    # Square root of NOT.
    cirq.ops.ExpWGate(half_turns=0.5).on(qubit),

    # Measurement.
    cirq.ops.MeasurementGate('result').on(qubit)
])
print("Circuit:")
print(cirq.circuits.to_ascii(circuit))

# Now simulate the circuit and print out the measurment result.
simulator = cirq.sim.google.xmon_simulator.Simulator()
results = []
for _ in range(10):
    result = simulator.run(circuit).measurements['result'][0]
    results.append('1' if result else '0')
print("Simulated measurement results:")
print(''.join(results))
```

## Documentation

See [here](docs/documentation.md).

## Contributing

We welcome contributions. Please follow [these](CONTRIBUTING) guidelines.
