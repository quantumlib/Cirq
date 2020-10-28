# Hello Qubit

A simple example to get you up and running:

```python
  import cirq

  # Pick a qubit.
  qubit = cirq.GridQubit(0, 0)

  # Create a circuit
  circuit = cirq.Circuit(
      cirq.X(qubit)**0.5,  # Square root of NOT.
      cirq.measure(qubit, key='m')  # Measurement.
  )
  print("Circuit:")
  print(circuit)

  # Simulate the circuit several times.
  simulator = cirq.Simulator()
  result = simulator.run(circuit, repetitions=20)
  print("Results:")
  print(result)
```

Example output:

```
  Circuit:
  (0, 0): ───X^0.5───M('m')───
  Results:
  m=11000111111011001000
```

# Learn more about quantum computing

In case you would like to learn more about Quantum Computing, check out our [education page](https://quantumai.google/education). 

# Hardware vs simulation

There are two main ways of running quantum algorithms using Cirq:
 - [Simulation](simulation.ipynb) is available on any computer
 - Quantum processors are provided by different Quantum Service Providers: 
    - [Google Quantum Computing Service](tutorials/google/start.ipynb)
    - [Alpine Quantum Technologies](tutorials/aqt/getting_started.ipynb)
    - [Pasqal](tutorials/pasqal/getting_started.ipynb) 
