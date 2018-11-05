.. currentmodule:: cirq

API Reference
=============

Circuits
--------

.. autosummary::
    :toctree: generated/

    Circuit
    Moment
    InsertStrategy
    OP_TREE


Operations
----------

.. autosummary::
    :toctree: generated/

    Operation
    GateOperation
    CompositeOperation
    QasmConvertibleOperation


Schedules
---------

.. autosummary::
    :toctree: generated/

    Schedule
    ScheduledOperation
    Duration
    Timestamp

Gates
^^^^^

.. autosummary::
    :toctree: generated/

    Gate
    MeasurementGate

Gate Features and Effects
'''''''''''''''''''''''''

.. autosummary::
    :toctree: generated

    CompositeGate
    ExtrapolatableEffect
    ReversibleEffect
    InterchangeableQubitsGate
    TextDiagrammable
    SingleQubitGate
    TwoQubitGate
    QasmConvertibleGate
    EigenGate

Single Qubit Gates
''''''''''''''''''

.. autosummary::
    :toctree: generated/

    RotXGate
    RotYGate
    RotZGate
    HGate
    X
    Y
    Z
    H
    S
    T


Two Qubit Gates
''''''''''''''''

.. autosummary::
    :toctree: generated/

    Rot11Gate
    CNotGate
    SwapGate
    ISwapGate
    CZ
    CNOT
    ISWAP

Three Qubit Gates
''''''''''''''''''

.. autosummary::
   :toctree: generated/

    CCZ
    CCX
    CSWAP
    TOFFOLI
    FREDKIN


Qubits
------

General classes for qubits and related concepts.

.. autosummary::
    :toctree: generated/

    QubitId
    NamedQubit
    LineQubit
    GridQubit
    QubitOrder
    QubitOrderOrList
    QubitOrder.DEFAULT


Devices
-------

Classes characterizing constraints of hardware.

.. autosummary::
    :toctree: generated/

    Device
    UnconstrainedDevice

Placement
---------

Classes for placing circuits onto circuits.

.. autosummary::
    :toctree: generated/

    LinePlacementStrategy
    GreedySequenceSearchStrategy
    AnnealSequenceSearchStrategy
    line_on_device

Parameterization
----------------

Classes for parameterized circuits.

.. autosummary::
    :toctree: generated/

    Symbol
    ParamResolver
    Sweep
    Points
    Linspace
    Sweepable

Optimization
------------

Classes for compiling.

.. autosummary::
    :toctree: generated/

    OptimizationPass
    PointOptimizer
    PointOptimizationSummary
    ExpandComposite
    DropEmptyMoments
    DropNegligible

Implementations
---------------

Packages to use specific hardware implementations.

Google
^^^^^^

Quantum hardware from Google.

Gates
''''''

.. autosummary::
    :toctree: generated/

    google.XmonGate
    google.Exp11Gate
    google.ExpWGate
    google.ExpZGate
    google.XmonMeasurementGate
    google.single_qubit_matrix_to_native_gates
    google.two_qubit_matrix_to_native_gates
    google.ConvertToXmonGates

Devices
'''''''

.. autosummary::
    :toctree: generated/

    google.Bristlecone
    google.Foxtail
    google.XmonDevice

Simulator
'''''''''

.. autosummary::
    :toctree: generated/

    google.XmonOptions
    google.XmonSimulator
    google.XmonStepResult
    google.XmonSimulateTrialResult

Optimizers
''''''''''

.. autosummary::
    :toctree: generated/

    google.optimized_for_xmon
    google.EjectZ
    google.EjectFullW
