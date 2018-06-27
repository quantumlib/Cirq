.. currentmodule:: cirq

API Reference
=============

Circuits
--------

.. autosummary::
    :toctree: generated/

    Circuit
    InsertStrategy
    Moment
    Schedule


Operations
----------

.. autosummary::
    :toctree: generated/

    Operation

Gates
^^^^^

.. autosummary::
    :toctree: generated/

    Gate
    MeasurementGate


Single Qubit Gates
''''''''''''''''''

.. autosummary::
    :toctree: generated/

    H
    X
    Y
    Z

Two Qubit Gates
''''''''''''''''

.. autosummary::
    :toctree: generated/

    CNOT
    CZ

Qubits
------

General classes for qubits.

.. autosummary::
    :toctree: generated/

    QubitId
    NamedQubit

See also:

* :ref:`Google Qubits <api-google-qubits>`


Implementations
---------------

Packages to use specific hardware implementations of quantum circuits.

Google
^^^^^^

Quantum hardware implementation by the Google Quantum AI Lab.

Engine
''''''

.. autosummary::
    :toctree: generated/

    google.Engine
    google.EngineOptions
    google.EngineTrialResult

Devices
'''''''

.. autosummary::
    :toctree: generated/

    google.Bristlecone
    google.Foxtail

Simulator
'''''''''

.. autosummary::
    :toctree: generated/

    google.Simulator
    google.Options
    google.StepResult
    google.TrialResult
    google.SimulatorTrialResult

.. _api-google-qubits:

Qubits
''''''

.. autosummary::
    :toctree: generated/

    google.XmonQubit
