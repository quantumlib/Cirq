.. image:: https://quantumai.google/site-assets/images/marketing/icons/ic-qcs.png
  :target: https://github.com/quantumlib/cirq/
  :alt: cirq-google
  :width: 500px

`Cirq <https://quantumai.google/cirq>`__ is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

This module is **cirq-google**, which provides everything you'll need to run Cirq quantum algorithms on the Google Quantum Computing Service.
It also contains additional tools for calibration and characterization of the Google quantum devices.

Documentation
-------------

Access to Google Hardware is currently restricted to those in an approved group. In order to do this, you will need to apply for access, typically in partnership with a Google sponsor.

To get started with the Quantum Computing Service, checkout the following guide and tutorial:

- `Access and authentication <https://quantumai.google/cirq/google/access>`__
- `Getting started guide <https://quantumai.google/cirq/tutorials/google/start>`__

Installation
------------

To install the stable version of only **cirq-google**, use `pip install cirq-google`.
To install the pre-release version of only **cirq-google**, use `pip install cirq-google~=1.0.dev`.

Note, that this will install both cirq-google and cirq-core as well.

To get all the optional modules installed as well, you'll have to use `pip install cirq` or `pip install cirq~=1.0.dev` for the pre-release version.
