.. image:: _static/Cirq_logo_color.png
    :alt: Cirq

Cirq
====

Cirq is a software library for writing, manipulating, and optimizing quantum
circuits and then running them against quantum computers and simulators.
Cirq attempts to expose the details of hardware, instead of abstracting them
away, because, in the Noisy Intermediate-Scale Quantum (NISQ) regime, these
details determine whether or not it is possible to execute a circuit at all.

Alpha Disclaimer
----------------

**Cirq is currently in alpha.**
We may change or remove parts of Cirq's API when making new releases.
To be informed of deprecations and breaking changes, subscribe to the
`cirq-announce google group mailing list <https://groups.google.com/forum/#!forum/cirq-announce>`__.

User Documentation
------------------

.. toctree::
    :maxdepth: 2

    install
    tutorial
    circuits
    gates
    noise
    simulation
    devices
    qudits
    development
    examples

Developer Documentation
-----------------------

.. toctree::
    :maxdepth: 1

    dev/index.rst


API Reference
-------------

.. toctree::
    :maxdepth: 2

    api

Google Documentation
-----------------------

.. toctree::
    :maxdepth: 1

    google/index.rst