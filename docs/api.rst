.. currentmodule:: cirq

API Reference
=============

Operations
----------

Gates
~~~~~

General Gates
'''''''''''''

.. autosummary::
    :toctree: generated/

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

Implementations
---------------

Packages to use specific hardware implementations of quantum circuits.

Google
~~~~~~

Quantum hardware implementation by the Google Quantum AI Lab.

Engine
''''''

.. autosummary::
    :toctree: generated/

    google.Engine
    google.JobConfig

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

    google.XmonOptions
    google.XmonSimulator
    google.XmonStepResult
    google.XmonSimulateTrialResult

Qubits
''''''

.. autosummary::
    :toctree: generated/

    google.XmonQubit
