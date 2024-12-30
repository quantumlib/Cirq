.. |cirqlogo| image:: https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg
   :alt: Cirq logo
   :target: https://github.com/quantumlib/cirq
   :height: 100px

.. |cirq| replace:: Cirq
.. _cirq: https://github.com/quantumlib/cirq

.. |cirq-docs| replace:: Cirq documentation site
.. _cirq-docs: https://quantumai.google/cirq

.. |cirq-github| replace:: Cirq GitHub repository
.. _cirq-github: https://github.com/quantumlib/Cirq

.. |cirq-releases| replace:: Cirq releases page
.. _cirq-releases: https://github.com/quantumlib/Cirq/releases

.. |cirq-core| replace:: ``cirq-core``

.. class:: centered

|cirqlogo|

|cirq|_ is a Python package for writing, manipulating, and running `quantum
circuits <https://en.wikipedia.org/wiki/Quantum_circuit>`__ on quantum
computers and simulators. Cirq provides useful abstractions for dealing with
today’s `noisy intermediate-scale quantum <https://arxiv.org/abs/1801.00862>`__
(NISQ) computers, where the details of quantum hardware are vital to achieving
state-of-the-art results. For more information about Cirq, please visit the
|cirq-docs|_.

This Python module is |cirq-core|, which contains everything you'd need to
write quantum algorithms for NISQ devices and run them on the built-in Cirq
simulators.

To run algorithms on a given quantum computing platform, you will also need to
install an appropriate Cirq hardware interface module. Please visit `the
hardware section of the Cirq documentation site
<https://quantumai.google/cirq/hardware>`_ for information about the hardware
interface modules currently available.


Installation
------------

There are two installation options for the |cirq-core| module:

* To install the stable version of |cirq-core|, use ``pip install cirq-core``.

* To install the pre-release version of |cirq-core|, use ``pip install
  cirq-core~=1.0.dev``. (The ``~=`` has a special meaning to ``pip`` of
  selecting the latest version compatible with the ``1.*`` and ``dev`` in the
  name. Despite apperances, this will not install an old version 1.0 release!)

If you would like to install Cirq with all the optional modules, not just
|cirq-core|, then instead of the above commands, use ``pip install cirq`` for
the stable release or ``pip install cirq~=1.0.dev`` for the latest pre-release
version.


Documentation
-------------

To get started with using Cirq, please refer to the |cirq-docs|_.

For more information about getting help, reporting bugs, and other matters
related to Cirq and the Cirq-Core integration module, please visit the
|cirq-github|_.


Disclaimer
----------

Cirq is not an official Google product. Copyright 2019 The Cirq Developers
