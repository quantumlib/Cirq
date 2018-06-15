# Cirq

[![Build Status](https://travis-ci.com/quantumlib/Cirq.svg?token=7FwHBHqoxBzvgH51kThw&branch=master)](https://travis-ci.com/quantumlib/Cirq)

Cirq is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

## Installation

Follow these [instructions](docs/install.md).

## Hello Qubit

A simple example to get you up and running:

```python
import cirq

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
```

Example output:

```
Circuit:
(0, 0): ───X^0.5───M───
Results:
m=11000111111011001000
```

## Documentation

See [here](docs/table_of_contents.md) or jump straight to the
[tutorial](docs/tutorial.md)

## Contributing

We welcome contributions. Please follow these [guidelines](CONTRIBUTING.md).

## Disclaimer

Copyright 2018 The Cirq Developers. This is not an official Google product.
