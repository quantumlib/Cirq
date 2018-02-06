# Cirq

Cirq is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

## Installation

Follow these [instructions](docs/install.md).

## Hello Qubit

A simple example to get you up and running:
```python
from cirq import ops
from cirq.circuits.circuit import Circuit
from cirq.sim.google.xmon_simulator import Simulator

# Define a qubit.
qubit = ops.QubitLoc(0, 0)

# Define moments, a sequence of gates.
moments = [Moment([sqrt_x_gate(qubit)]), Moment([meas(qubit)])]

# Define a square root of not gate and a measurement gate.
sqrt_x_gate = ops.ExpWGate(half_turns=0.25)
meas = ops.MeasurementGate('result')

# Create a circuit (qubit will start in |0> state).
circuit = Circuit(moments)

# Now simulate the circuit and print out the measurment result.
simulator = Simulator()
results = simulator.run(circuit)
print('Measurement result: %s' % results.measurements['result'])
```

## Documentation

See [here](docs/documentation.md).

## Contributing

We welcome contributions. Please follow [these](CONTRIBUTING) guidelines.
