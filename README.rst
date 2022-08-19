.. image:: https://raw.githubusercontent.com/quantumlib/Cirq/master/docs/images/Cirq_logo_color.png
  :target: https://github.com/quantumlib/cirq
  :alt: Cirq
  :width: 500px

Cirq is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

.. image:: https://github.com/quantumlib/Cirq/actions/workflows/ci.yml/badge.svg?event=schedule
  :target: https://github.com/quantumlib/Cirq
  :alt: Build Status

.. image:: https://badge.fury.io/py/cirq.svg
  :target: https://badge.fury.io/py/cirq

.. image:: https://readthedocs.org/projects/cirq/badge/?version=latest
  :target: https://readthedocs.org/projects/cirq/versions/
  :alt: Documentation Status


Installation and Documentation
------------------------------

Cirq documentation is available at `quantumai.google/cirq <https://quantumai.google/cirq>`_.

Documentation for the latest **pre-release** version of cirq (tracks the repository's master branch; what you get if you ``pip install --pre cirq``), is available `here <https://quantumai.google/reference/python/cirq/all_symbols?version=nightly>`__.

Documentation for the latest **stable** version of cirq (what you get if you ``pip install cirq``) is available `here <https://quantumai.google/reference/python/cirq/all_symbols>`__.


- `Installation <https://quantumai.google/cirq/start/install>`_
- `Documentation <https://quantumai.google/cirq>`_
- `Tutorials <https://quantumai.google/cirq/build>`_

For a comprehensive list all of the interactive Jupyter Notebooks in our repo (including the ones not yet published to the site) open our repo in `Colab <https://colab.research.google.com/github/quantumlib/Cirq>`_.

For the latest news regarding Cirq, sign up to the `Cirq-announce email list <https://groups.google.com/forum/#!forum/cirq-announce>`__!


Hello Qubit
-----------

A simple example to get you up and running:

.. code-block:: python

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

Example output:

.. code-block::

  Circuit:
  (0, 0): ───X^0.5───M('m')───
  Results:
  m=11000111111011001000


Feature requests / Bugs / Questions
-----------------------------------

If you have feature requests or you found a bug, please `file them on Github <https://github.com/quantumlib/Cirq/issues/new/choose>`__.

For questions about how to use Cirq post to
`Quantum Computing Stack Exchange <https://quantumcomputing.stackexchange.com/>`__ with the
`cirq <https://quantumcomputing.stackexchange.com/questions/tagged/cirq>`__ tag.

How to cite Cirq
----------------

Cirq is uploaded to Zenodo automatically. Click on the badge below to see all the citation formats for all versions.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4062499.svg
  :target: https://doi.org/10.5281/zenodo.4062499
  :alt: DOI

Cirq Contributors Community
---------------------------

We welcome contributions! Before opening your first PR, a good place to start is to read our
`guidelines <https://github.com/quantumlib/cirq/blob/master/CONTRIBUTING.md>`__.

We are dedicated to cultivating an open and inclusive community to build software for near term quantum computers.
Please read our `code of conduct <https://github.com/quantumlib/cirq/blob/master/CODE_OF_CONDUCT.md>`__ for the rules of engagement within our community.

For real time informal discussions about Cirq, join our `cirqdev <https://gitter.im/cirqdev>`__ Gitter channel, come hangout with us!

**Cirq Cynque** is our weekly meeting for contributors to discuss upcoming features, designs, issues, community and status of different efforts.
To get an invitation please join the `cirq-dev email list <https://groups.google.com/forum/#!forum/cirq-dev>`__ which also serves as yet another platform to discuss contributions and design ideas.


See Also
--------

For those interested in using quantum computers to solve problems in
chemistry and materials science, we encourage exploring
`OpenFermion <https://github.com/quantumlib/openfermion>`__ and
its sister library for compiling quantum simulation algorithms in Cirq,
`OpenFermion-Cirq <https://github.com/quantumlib/openfermion-cirq>`__.

For machine learning enthusiasts, `Tensorflow Quantum <https://github.com/tensorflow/quantum>`__ is a great project to check out!

For a powerful quantum circuit simulator that integrates well with Cirq, we recommend looking at `qsim <https://github.com/quantumlib/qsim>`__.

Finally, `ReCirq <https://github.com/quantumlib/ReCirq>`__ contains real world experiments using Cirq.

Cirq is not an official Google product. Copyright 2019 The Cirq Developers
