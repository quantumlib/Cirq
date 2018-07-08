Cirq
====

.. image:: https://travis-ci.com/quantumlib/Cirq.svg?token=7FwHBHqoxBzvgH51kThw&branch=master
  :target: https://travis-ci.com/quantumlib/Cirq
  :alt: Build Status

Cirq is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

Installation
------------

Follow these
`instructions <https://github.com/quantumlib/Cirq/blob/master/docs/install.md>`__.

Hello Qubit
-----------

A simple example to get you up and running:

.. code-block:: python

  import cirq

  # Pick a qubit.
  qubit = cirq.devices.GridQubit(0, 0)

  # Create a circuit
  circuit = cirq.Circuit.from_ops(
      cirq.X(qubit)**0.5,  # Square root of NOT.
      cirq.measure(qubit, key='m')  # Measurement.
  )
  print("Circuit:")
  print(circuit)

  # Simulate the circuit several times.
  simulator = cirq.google.XmonSimulator()
  result = simulator.run(circuit, repetitions=20)
  print("Results:")
  print(result)

Example output:

.. code-block:: bash

  Circuit:
  (0, 0): ───X^0.5───M───
  Results:
  m=11000111111011001000


Documentation
-------------

See
`here <https://github.com/quantumlib/Cirq/blob/master/docs/table_of_contents.md>`__
or jump into the
`tutorial <https://github.com/quantumlib/Cirq/blob/master/docs/tutorial.md>`__.

Contributing
------------

We welcome contributions. Please follow these
`guidelines <https://github.com/quantumlib/cirq/blob/master/CONTRIBUTING.md>`__.

Disclaimer
----------

Copyright 2018 The Cirq Developers. This is not an official Google product.
