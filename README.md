# Cirq

Cirq is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

## Installation

Follow these [instructions](cirq/docs/install.md).

## Hello Qubit

A simple example to get you up and running:

```python
import cirq

# Define a qubit.
qubit = cirq.google.XmonQubit(0, 0)

# Create a circuit (qubits start in the |0> state).
circuit = cirq.Circuit()
circuit.append([
    # Square root of NOT.
    cirq.X.on(qubit)**0.5,

    # Measurement.
    cirq.MeasurementGate('result').on(qubit)
])
print("Circuit:")
print(circuit)

# Now simulate the circuit and print out the measurement result.
simulator = cirq.google.Simulator()
result = simulator.run(circuit, repetitions=10)
results = [str(int(b)) for b in result.measurements['result'][:, 0]]
print("Simulated measurement results:")
print(''.join(results))
```

Example output:

```
Circuit:
(0, 0): ───X^0.5───M───

Simulated measurement results:
1110111010
```

## Documentation

See [here](cirq/docs/table_of_contents.md) or jump straight to the
[tutorial](cirq/docs/tutorial.md)

## Contributing

We welcome contributions. Please follow these [guidelines](CONTRIBUTING.md).

## Disclaimer

Copyright 2018 The Cirq Developers. This is not an official Google product.
