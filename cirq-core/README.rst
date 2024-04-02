.. image:: https://raw.githubusercontent.com/quantumlib/Cirq/main/docs/images/Cirq_logo_color.png
  :target: https://github.com/quantumlib/cirq
  :alt: Cirq
  :width: 500px

Cirq is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

This module is **cirq-core**, which contains everything you'd need to write quantum algorithms for NISQ devices and run them on the built-in Cirq simulators.
In order to run algorithms on a given quantum hardware platform, you'll have to install the right cirq module as well.

Installation
------------

To install the stable version of only **cirq-core**, use `pip install cirq-core`.
To install the pre-release version of only **cirq-core**, use `pip install cirq-core~=1.0.dev`.

To get all the optional modules installed as well, you'll have to use `pip install cirq` or `pip install cirq~=1.0.dev` for the pre-release version.
