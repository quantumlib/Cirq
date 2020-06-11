<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="CONTROL_TAG"/>
<meta itemprop="property" content="DEFAULT_RESOLVERS"/>
<meta itemprop="property" content="NO_NOISE"/>
<meta itemprop="property" content="PAULI_BASIS"/>
<meta itemprop="property" content="UNCONSTRAINED_DEVICE"/>
<meta itemprop="property" content="UnitSweep"/>
<meta itemprop="property" content="__version__"/>
</div>

# Module: cirq

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Modules

[`circuits`](./cirq/circuits.md) module: Types and methods related to building and optimizing sequenced circuits.

[`contrib`](./cirq/contrib.md) module: Package for contributions.

[`devices`](./cirq/devices.md) module

[`experiments`](./cirq/experiments.md) module

[`google`](./cirq/google.md) module

[`interop`](./cirq/interop.md) module: Package containing code for interoperating with other quantum software.

[`ion`](./cirq/ion.md) module: Types for representing and methods for manipulating ion trap operations.

[`linalg`](./cirq/linalg.md) module: Types and methods related to performing linear algebra.

[`neutral_atoms`](./cirq/neutral_atoms.md) module

[`ops`](./cirq/ops.md) module: Types for representing and methods for manipulating circuit operation trees.

[`optimizers`](./cirq/optimizers.md) module: Circuit transformation utilities.

[`pasqal`](./cirq/pasqal.md) module

[`protocols`](./cirq/protocols.md) module

[`qis`](./cirq/qis.md) module

[`sim`](./cirq/sim.md) module: Base simulation classes and generic simulators.

[`study`](./cirq/study.md) module: Types and methods for running studies (repeated trials).

[`testing`](./cirq/testing.md) module: Utilities for testing code.

[`type_workarounds`](./cirq/type_workarounds.md) module

[`value`](./cirq/value.md) module

[`vis`](./cirq/vis.md) module

[`work`](./cirq/work.md) module

## Classes

[`class ABCMetaImplementAnyOneOf`](./cirq/value/ABCMetaImplementAnyOneOf.md): A metaclass extending `abc.ABCMeta` for defining abstract base classes

[`class ActOnStateVectorArgs`](./cirq/sim/ActOnStateVectorArgs.md): State and context for an operation acting on a state vector.

[`class AmplitudeDampingChannel`](./cirq/ops/AmplitudeDampingChannel.md): Dampen qubit amplitudes through dissipation.

[`class ApplyChannelArgs`](./cirq/protocols/ApplyChannelArgs.md): Arguments for efficiently performing a channel.

[`class ApplyMixtureArgs`](./cirq/protocols/ApplyMixtureArgs.md): Arguments for performing a mixture of unitaries.

[`class ApplyUnitaryArgs`](./cirq/protocols/ApplyUnitaryArgs.md): Arguments for performing an efficient left-multiplication by a unitary.

[`class ArithmeticOperation`](./cirq/ops/ArithmeticOperation.md): A helper class for implementing reversible classical arithmetic.

[`class AsymmetricDepolarizingChannel`](./cirq/ops/AsymmetricDepolarizingChannel.md): A channel that depolarizes asymmetrically along different directions.

[`class AxisAngleDecomposition`](./cirq/linalg/AxisAngleDecomposition.md): Represents a unitary operation as an axis, angle, and global phase.

[`class BaseDensePauliString`](./cirq/ops/BaseDensePauliString.md): Parent class for `DensePauliString` and `MutableDensePauliString`.

[`class BitFlipChannel`](./cirq/ops/BitFlipChannel.md): Probabilistically flip a qubit from 1 to 0 state or vice versa.

[`class CCNotPowGate`](./cirq/ops/CCNotPowGate.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`class CCXPowGate`](./cirq/ops/CCNotPowGate.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`class CCZPowGate`](./cirq/ops/CCZPowGate.md): A doubly-controlled-Z that can be raised to a power.

[`class CNotPowGate`](./cirq/ops/CNotPowGate.md): A gate that applies a controlled power of an X gate.

[`class CSwapGate`](./cirq/ops/CSwapGate.md): A controlled swap gate. The Fredkin gate.

[`class CXPowGate`](./cirq/ops/CNotPowGate.md): A gate that applies a controlled power of an X gate.

[`class CZPowGate`](./cirq/ops/CZPowGate.md): A gate that applies a phase to the |11⟩ state of two qubits.

[`class Circuit`](./cirq/circuits/Circuit.md): A mutable list of groups of operations to apply to some qubits.

[`class CircuitDag`](./cirq/circuits/CircuitDag.md): A representation of a Circuit as a directed acyclic graph.

[`class CircuitDiagramInfo`](./cirq/protocols/CircuitDiagramInfo.md): Describes how to draw an operation in a circuit diagram.

[`class CircuitDiagramInfoArgs`](./cirq/protocols/CircuitDiagramInfoArgs.md): A request for information on drawing an operation in a circuit diagram.

[`class CircuitSampleJob`](./cirq/work/CircuitSampleJob.md): Describes a sampling task.

[`class CliffordSimulator`](./cirq/sim/CliffordSimulator.md): An efficient simulator for Clifford circuits.

[`class CliffordSimulatorStepResult`](./cirq/sim/CliffordSimulatorStepResult.md): A `StepResult` that includes `StateVectorMixin` methods.

[`class CliffordState`](./cirq/sim/CliffordState.md): A state of the Clifford simulation.

[`class CliffordTableau`](./cirq/sim/CliffordTableau.md): Tableau representation of a stabilizer state

[`class CliffordTrialResult`](./cirq/sim/CliffordTrialResult.md): Results of a simulation by a SimulatesFinalState.

[`class Collector`](./cirq/work/Collector.md): Collects data from a sampler, in parallel, towards some purpose.

[`class ConstantQubitNoiseModel`](./cirq/devices/ConstantQubitNoiseModel.md): Applies noise to each qubit individually at the start of every moment.

[`class ControlledGate`](./cirq/ops/ControlledGate.md): Augments existing gates to have one or more control qubits.

[`class ControlledOperation`](./cirq/ops/ControlledOperation.md): Augments existing operations to have one or more control qubits.

[`class ConvertToCzAndSingleGates`](./cirq/optimizers/ConvertToCzAndSingleGates.md): Attempts to convert strange multi-qubit gates into CZ and single qubit

[`class ConvertToIonGates`](./cirq/ion/ConvertToIonGates.md): Attempts to convert non-native gates into IonGates.

[`class ConvertToNeutralAtomGates`](./cirq/neutral_atoms/ConvertToNeutralAtomGates.md): Attempts to convert gates into native Atom gates.

[`class DensePauliString`](./cirq/ops/DensePauliString.md): Parent class for `DensePauliString` and `MutableDensePauliString`.

[`class DensityMatrixSimulator`](./cirq/sim/DensityMatrixSimulator.md): A simulator for density matrices and noisy quantum circuits.

[`class DensityMatrixSimulatorState`](./cirq/sim/DensityMatrixSimulatorState.md): The simulator state for DensityMatrixSimulator

[`class DensityMatrixStepResult`](./cirq/sim/DensityMatrixStepResult.md): A single step in the simulation of the DensityMatrixSimulator.

[`class DensityMatrixTrialResult`](./cirq/sim/DensityMatrixTrialResult.md): A `SimulationTrialResult` for `DensityMatrixSimulator` runs.

[`class DepolarizingChannel`](./cirq/ops/DepolarizingChannel.md): A channel that depolarizes a qubit.

[`class Device`](./cirq/devices/Device.md): Hardware constraints for validating circuits.

[`class DropEmptyMoments`](./cirq/optimizers/DropEmptyMoments.md): Removes empty moments from a circuit.

[`class DropNegligible`](./cirq/optimizers/DropNegligible.md): An optimization pass that removes operations with tiny effects.

[`class Duration`](./cirq/value/Duration.md): A time delta that supports symbols and picosecond accuracy.

[`class EigenGate`](./cirq/ops/EigenGate.md): A gate with a known eigendecomposition.

[`class EjectPhasedPaulis`](./cirq/optimizers/EjectPhasedPaulis.md): Pushes X, Y, and PhasedX gates towards the end of the circuit.

[`class EjectZ`](./cirq/optimizers/EjectZ.md): Pushes Z gates towards the end of the circuit.

[`class ExpandComposite`](./cirq/optimizers/ExpandComposite.md): An optimizer that expands composite operations via <a href="./cirq/protocols/decompose.md"><code>cirq.decompose</code></a>.

[`class ExpressionMap`](./cirq/study/ExpressionMap.md): A dictionary with sympy expressions and symbols for keys and sympy

[`class FSimGate`](./cirq/ops/FSimGate.md): Fermionic simulation gate family.

[`class Gate`](./cirq/ops/Gate.md): An operation type that can be applied to a collection of qubits.

[`class GateOperation`](./cirq/ops/GateOperation.md): An application of a gate to a sequence of qubits.

[`class GeneralizedAmplitudeDampingChannel`](./cirq/ops/GeneralizedAmplitudeDampingChannel.md): Dampen qubit amplitudes through non ideal dissipation.

[`class GlobalPhaseOperation`](./cirq/ops/GlobalPhaseOperation.md): An effect applied to a collection of qubits.

[`class GridQid`](./cirq/devices/GridQid.md): A qid on a 2d square lattice

[`class GridQubit`](./cirq/devices/GridQubit.md): A qubit on a 2d square lattice.

[`class HPowGate`](./cirq/ops/HPowGate.md): A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

[`class Heatmap`](./cirq/vis/Heatmap.md): Distribution of a value in 2D qubit lattice as a color map.

[`class ISwapPowGate`](./cirq/ops/ISwapPowGate.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`class IdentityGate`](./cirq/ops/IdentityGate.md): A Gate that perform no operation on qubits.

[`class InsertStrategy`](./cirq/circuits/InsertStrategy.md): Indicates preferences on how to add multiple operations to a circuit.

[`class InterchangeableQubitsGate`](./cirq/ops/InterchangeableQubitsGate.md): Indicates operations should be equal under some qubit permutations.

[`class IonDevice`](./cirq/ion/IonDevice.md): A device with qubits placed on a line.

[`class KakDecomposition`](./cirq/linalg/KakDecomposition.md): A convenient description of an arbitrary two-qubit operation.

[`class LineQid`](./cirq/devices/LineQid.md): A qid on a 1d lattice with nearest-neighbor connectivity.

[`class LineQubit`](./cirq/devices/LineQubit.md): A qubit on a 1d lattice with nearest-neighbor connectivity.

[`class LinearCombinationOfGates`](./cirq/ops/LinearCombinationOfGates.md): Represents linear operator defined by a linear combination of gates.

[`class LinearCombinationOfOperations`](./cirq/ops/LinearCombinationOfOperations.md): Represents operator defined by linear combination of gate operations.

[`class LinearDict`](./cirq/value/LinearDict.md): Represents linear combination of things.

[`class Linspace`](./cirq/study/Linspace.md): A simple sweep over linearly-spaced values.

[`class ListSweep`](./cirq/study/ListSweep.md): A wrapper around a list of `ParamResolver`s.

[`class MatrixGate`](./cirq/ops/MatrixGate.md): A unitary qubit or qudit gate defined entirely by its matrix.

[`class MeasurementGate`](./cirq/ops/MeasurementGate.md): A gate that measures qubits in the computational basis.

[`class MergeInteractions`](./cirq/optimizers/MergeInteractions.md): Combines series of adjacent one and two-qubit gates operating on a pair

[`class MergeSingleQubitGates`](./cirq/optimizers/MergeSingleQubitGates.md): Optimizes runs of adjacent unitary 1-qubit operations.

[`class Moment`](./cirq/ops/Moment.md): A time-slice of operations within a circuit.

[`class MutableDensePauliString`](./cirq/ops/MutableDensePauliString.md): Parent class for `DensePauliString` and `MutableDensePauliString`.

[`class NamedQubit`](./cirq/ops/NamedQubit.md): A qubit identified by name.

[`class NeutralAtomDevice`](./cirq/neutral_atoms/NeutralAtomDevice.md): A device with qubits placed on a grid.

[`class NoiseModel`](./cirq/devices/NoiseModel.md): Replaces operations and moments with noisy counterparts.

[`class Operation`](./cirq/ops/Operation.md): An effect applied to a collection of qubits.

[`class ParallelGateOperation`](./cirq/ops/ParallelGateOperation.md): An application of several copies of a gate to a group of qubits.

[`class ParamResolver`](./cirq/study/ParamResolver.md): Resolves sympy.Symbols to actual values.

[`class Pauli`](./cirq/ops/Pauli.md): Represents the Pauli gates.

[`class PauliInteractionGate`](./cirq/ops/PauliInteractionGate.md): A CZ conjugated by arbitrary single qubit Cliffords.

[`class PauliString`](./cirq/ops/PauliString.md): An effect applied to a collection of qubits.

[`class PauliStringGateOperation`](./cirq/ops/PauliStringGateOperation.md): An effect applied to a collection of qubits.

[`class PauliStringPhasor`](./cirq/ops/PauliStringPhasor.md): An operation that phases the eigenstates of a Pauli string.

[`class PauliSum`](./cirq/ops/PauliSum.md): Represents operator defined by linear combination of PauliStrings.

[`class PauliSumCollector`](./cirq/work/PauliSumCollector.md): Estimates the energy of a linear combination of Pauli observables.

[`class PauliTransform`](./cirq/ops/PauliTransform.md): PauliTransform(to, flip)

[`class PeriodicValue`](./cirq/value/PeriodicValue.md): Wrapper for periodic numerical values.

[`class PhaseDampingChannel`](./cirq/ops/PhaseDampingChannel.md): Dampen qubit phase.

[`class PhaseFlipChannel`](./cirq/ops/PhaseFlipChannel.md): Probabilistically flip the sign of the phase of a qubit.

[`class PhaseGradientGate`](./cirq/ops/PhaseGradientGate.md): Phases each state |k⟩ out of n by e^(2*pi*i*k/n*exponent).

[`class PhasedISwapPowGate`](./cirq/ops/PhasedISwapPowGate.md): Fractional ISWAP conjugated by Z rotations.

[`class PhasedXPowGate`](./cirq/ops/PhasedXPowGate.md): A gate equivalent to the circuit ───Z^-p───X^t───Z^p───.

[`class PhasedXZGate`](./cirq/ops/PhasedXZGate.md): A single qubit operation expressed as $Z^z Z^a X^x Z^{-a}$.

[`class PointOptimizationSummary`](./cirq/circuits/PointOptimizationSummary.md): A description of a local optimization to perform.

[`class PointOptimizer`](./cirq/circuits/PointOptimizer.md): Makes circuit improvements focused on a specific location.

[`class Points`](./cirq/study/Points.md): A simple sweep with explicitly supplied values.

[`class Product`](./cirq/study/Product.md): Cartesian product of one or more sweeps.

[`class QasmArgs`](./cirq/protocols/QasmArgs.md)

[`class QasmOutput`](./cirq/circuits/QasmOutput.md)

[`class Qid`](./cirq/ops/Qid.md): Identifies a quantum object such as a qubit, qudit, resonator, etc.

[`class QuantumFourierTransformGate`](./cirq/ops/QuantumFourierTransformGate.md): Switches from the computational basis to the frequency basis.

[`class QubitOrder`](./cirq/ops/QubitOrder.md): Defines the kronecker product order of qubits.

[`class QuilFormatter`](./cirq/protocols/QuilFormatter.md): A unique formatter to correctly output values to QUIL.

[`class QuilOutput`](./cirq/circuits/QuilOutput.md): An object for passing operations and qubits then outputting them to

[`class RandomGateChannel`](./cirq/ops/RandomGateChannel.md): Applies a sub gate with some probability.

[`class ResetChannel`](./cirq/ops/ResetChannel.md): Reset a qubit back to its |0⟩ state.

[`class Sampler`](./cirq/work/Sampler.md): Something capable of sampling quantum circuits. Simulator or hardware.

[`class SimulatesAmplitudes`](./cirq/sim/SimulatesAmplitudes.md): Simulator that computes final amplitudes of given bitstrings.

[`class SimulatesFinalState`](./cirq/sim/SimulatesFinalState.md): Simulator that allows access to the simulator's final state.

[`class SimulatesIntermediateState`](./cirq/sim/SimulatesIntermediateState.md): A SimulatesFinalState that simulates a circuit by moments.

[`class SimulatesIntermediateStateVector`](./cirq/sim/SimulatesIntermediateStateVector.md): A simulator that accesses its state vector as it does its simulation.

[`class SimulatesIntermediateWaveFunction`](./cirq/sim/SimulatesIntermediateWaveFunction.md): Deprecated. Please use `SimulatesIntermediateStateVector` instead.

[`class SimulatesSamples`](./cirq/sim/SimulatesSamples.md): Simulator that mimics running on quantum hardware.

[`class SimulationTrialResult`](./cirq/sim/SimulationTrialResult.md): Results of a simulation by a SimulatesFinalState.

[`class Simulator`](./cirq/sim/Simulator.md): A sparse matrix state vector simulator that uses numpy.

[`class SingleQubitCliffordGate`](./cirq/ops/SingleQubitCliffordGate.md): Any single qubit Clifford rotation.

[`class SingleQubitGate`](./cirq/ops/SingleQubitGate.md): A gate that must be applied to exactly one qubit.

[`class SingleQubitPauliStringGateOperation`](./cirq/ops/SingleQubitPauliStringGateOperation.md): A Pauli operation applied to a qubit.

[`class SparseSimulatorStep`](./cirq/sim/SparseSimulatorStep.md): A `StepResult` that includes `StateVectorMixin` methods.

[`class StabilizerStateChForm`](./cirq/sim/StabilizerStateChForm.md): A representation of stabilizer states using the CH form,

[`class StateVectorMixin`](./cirq/sim/StateVectorMixin.md): A mixin that provide methods for objects that have a state vector.

[`class StateVectorSimulatorState`](./cirq/sim/StateVectorSimulatorState.md)

[`class StateVectorStepResult`](./cirq/sim/StateVectorStepResult.md): Results of a step of a SimulatesIntermediateState.

[`class StateVectorTrialResult`](./cirq/sim/StateVectorTrialResult.md): A `SimulationTrialResult` that includes the `StateVectorMixin` methods.

[`class StepResult`](./cirq/sim/StepResult.md): Results of a step of a SimulatesIntermediateState.

[`class SupportsActOn`](./cirq/protocols/SupportsActOn.md): An object that explicitly specifies how to act on simulator states.

[`class SupportsApplyChannel`](./cirq/protocols/SupportsApplyChannel.md): An object that can efficiently implement a channel.

[`class SupportsApplyMixture`](./cirq/protocols/SupportsApplyMixture.md): An object that can efficiently implement a mixture.

[`class SupportsApproximateEquality`](./cirq/protocols/SupportsApproximateEquality.md): Object which can be compared approximately.

[`class SupportsChannel`](./cirq/protocols/SupportsChannel.md): An object that may be describable as a quantum channel.

[`class SupportsCircuitDiagramInfo`](./cirq/protocols/SupportsCircuitDiagramInfo.md): A diagrammable operation on qubits.

[`class SupportsCommutes`](./cirq/protocols/SupportsCommutes.md): An object that can determine commutation relationships vs others.

[`class SupportsConsistentApplyUnitary`](./cirq/protocols/SupportsConsistentApplyUnitary.md): An object that can be efficiently left-multiplied into tensors.

[`class SupportsDecompose`](./cirq/protocols/SupportsDecompose.md): An object that can be decomposed into simpler operations.

[`class SupportsDecomposeWithQubits`](./cirq/protocols/SupportsDecomposeWithQubits.md): An object that can be decomposed into operations on given qubits.

[`class SupportsEqualUpToGlobalPhase`](./cirq/protocols/SupportsEqualUpToGlobalPhase.md): Object which can be compared for equality mod global phase.

[`class SupportsExplicitHasUnitary`](./cirq/protocols/SupportsExplicitHasUnitary.md): An object that explicitly specifies whether it has a unitary effect.

[`class SupportsExplicitNumQubits`](./cirq/protocols/SupportsExplicitNumQubits.md): A unitary, channel, mixture or other object that operates on a known

[`class SupportsExplicitQidShape`](./cirq/protocols/SupportsExplicitQidShape.md): A unitary, channel, mixture or other object that operates on a known

[`class SupportsJSON`](./cirq/protocols/SupportsJSON.md): An object that can be turned into JSON dictionaries.

[`class SupportsMeasurementKey`](./cirq/protocols/SupportsMeasurementKey.md): An object that is a measurement and has a measurement key or keys.

[`class SupportsMixture`](./cirq/protocols/SupportsMixture.md): An object that decomposes into a probability distribution of unitaries.

[`class SupportsParameterization`](./cirq/protocols/SupportsParameterization.md): An object that can be parameterized by Symbols and resolved

[`class SupportsPhase`](./cirq/protocols/SupportsPhase.md): An effect that can be phased around the Z axis of target qubits.

[`class SupportsQasm`](./cirq/protocols/SupportsQasm.md): An object that can be turned into QASM code.

[`class SupportsQasmWithArgs`](./cirq/protocols/SupportsQasmWithArgs.md): An object that can be turned into QASM code.

[`class SupportsQasmWithArgsAndQubits`](./cirq/protocols/SupportsQasmWithArgsAndQubits.md): An object that can be turned into QASM code if it knows its qubits.

[`class SupportsTraceDistanceBound`](./cirq/protocols/SupportsTraceDistanceBound.md): An effect with known bounds on how easy it is to detect.

[`class SupportsUnitary`](./cirq/protocols/SupportsUnitary.md): An object that may be describable by a unitary matrix.

[`class SwapPowGate`](./cirq/ops/SwapPowGate.md): The SWAP gate, possibly raised to a power. Exchanges qubits.

[`class Sweep`](./cirq/study/Sweep.md): A sweep is an iterator over ParamResolvers.

[`class SynchronizeTerminalMeasurements`](./cirq/optimizers/SynchronizeTerminalMeasurements.md): Move measurements to the end of the circuit.

[`class TaggedOperation`](./cirq/ops/TaggedOperation.md): A specific operation instance that has been identified with a set

[`class TextDiagramDrawer`](./cirq/circuits/TextDiagramDrawer.md): A utility class for creating simple text diagrams.

[`class ThreeQubitDiagonalGate`](./cirq/ops/ThreeQubitDiagonalGate.md): A gate given by a diagonal 8x8 matrix.

[`class ThreeQubitGate`](./cirq/ops/ThreeQubitGate.md): A gate that must be applied to exactly three qubits.

[`class Timestamp`](./cirq/value/Timestamp.md): A location in time with picosecond accuracy.

[`class TrialResult`](./cirq/study/TrialResult.md): The results of multiple executions of a circuit with fixed parameters.

[`class TwoQubitDiagonalGate`](./cirq/ops/TwoQubitDiagonalGate.md): A gate given by a diagonal 4\times 4 matrix.

[`class TwoQubitGate`](./cirq/ops/TwoQubitGate.md): A gate that must be applied to exactly two qubits.

[`class Unique`](./cirq/circuits/Unique.md): A wrapper for a value that doesn't compare equal to other instances.

[`class VirtualTag`](./cirq/ops/VirtualTag.md): A TaggedOperation tag indicating that the operation is virtual.

[`class WaitGate`](./cirq/ops/WaitGate.md): A single-qubit idle gate that represents waiting.

[`class WaveFunctionSimulatorState`](./cirq/sim/WaveFunctionSimulatorState.md): Deprecated. Please use `StateVectorSimulatorState` instead.

[`class WaveFunctionStepResult`](./cirq/sim/WaveFunctionStepResult.md): Deprecated. Please use `StateVectorStepResult` instead.

[`class WaveFunctionTrialResult`](./cirq/sim/WaveFunctionTrialResult.md): Deprecated. Please use `StateVectorTrialResult` instead.

[`class XPowGate`](./cirq/ops/XPowGate.md): A gate that rotates around the X axis of the Bloch sphere.

[`class XXPowGate`](./cirq/ops/XXPowGate.md): The X-parity gate, possibly raised to a power.

[`class YPowGate`](./cirq/ops/YPowGate.md): A gate that rotates around the Y axis of the Bloch sphere.

[`class YYPowGate`](./cirq/ops/YYPowGate.md): The Y-parity gate, possibly raised to a power.

[`class ZPowGate`](./cirq/ops/ZPowGate.md): A gate that rotates around the Z axis of the Bloch sphere.

[`class ZZPowGate`](./cirq/ops/ZZPowGate.md): The Z-parity gate, possibly raised to a power.

[`class Zip`](./cirq/study/Zip.md): Zip product (direct sum) of one or more sweeps.

## Functions

[`CCNOT(...)`](./cirq/ops/CCNOT.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`CCX(...)`](./cirq/ops/CCNOT.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`CCZ(...)`](./cirq/ops/CCZ.md): A doubly-controlled-Z that can be raised to a power.

[`CNOT(...)`](./cirq/ops/CNOT.md): A gate that applies a controlled power of an X gate.

[`CSWAP(...)`](./cirq/ops/CSWAP.md): A controlled swap gate. The Fredkin gate.

[`CX(...)`](./cirq/ops/CNOT.md): A gate that applies a controlled power of an X gate.

[`CZ(...)`](./cirq/ops/CZ.md): A gate that applies a phase to the |11⟩ state of two qubits.

[`FREDKIN(...)`](./cirq/ops/CSWAP.md): A controlled swap gate. The Fredkin gate.

[`H(...)`](./cirq/ops/H.md): A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

[`I(...)`](./cirq/ops/I.md): A Gate that perform no operation on qubits.

[`ISWAP(...)`](./cirq/ops/ISWAP.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`QFT(...)`](./cirq/ops/QFT.md): THIS FUNCTION IS DEPRECATED.

[`S(...)`](./cirq/ops/S.md): A gate that rotates around the Z axis of the Bloch sphere.

[`SWAP(...)`](./cirq/ops/SWAP.md): The SWAP gate, possibly raised to a power. Exchanges qubits.

[`T(...)`](./cirq/ops/T.md): A gate that rotates around the Z axis of the Bloch sphere.

[`TOFFOLI(...)`](./cirq/ops/CCNOT.md): A Toffoli (doubly-controlled-NOT) that can be raised to a power.

[`X(...)`](./cirq/ops/X.md)

[`XX(...)`](./cirq/ops/XX.md): The X-parity gate, possibly raised to a power.

[`Y(...)`](./cirq/ops/Y.md)

[`YY(...)`](./cirq/ops/YY.md): The Y-parity gate, possibly raised to a power.

[`Z(...)`](./cirq/ops/Z.md)

[`ZZ(...)`](./cirq/ops/ZZ.md): The Z-parity gate, possibly raised to a power.

[`act_on(...)`](./cirq/protocols/act_on.md): Applies an action to a state argument.

[`all_near_zero(...)`](./cirq/linalg/all_near_zero.md): Checks if the tensor's elements are all near zero.

[`all_near_zero_mod(...)`](./cirq/linalg/all_near_zero_mod.md): Checks if the tensor's elements are all near multiples of the period.

[`allclose_up_to_global_phase(...)`](./cirq/linalg/allclose_up_to_global_phase.md): Determines if a ~= b * exp(i t) for some t.

[`alternative(...)`](./cirq/value/alternative.md): A decorator indicating an abstract method with an alternative default

[`amplitude_damp(...)`](./cirq/ops/amplitude_damp.md): Returns an AmplitudeDampingChannel with the given probability gamma.

[`apply_channel(...)`](./cirq/protocols/apply_channel.md): High performance evolution under a channel evolution.

[`apply_matrix_to_slices(...)`](./cirq/linalg/apply_matrix_to_slices.md): Left-multiplies an NxN matrix onto N slices of a numpy array.

[`apply_mixture(...)`](./cirq/protocols/apply_mixture.md): High performance evolution under a mixture of unitaries evolution.

[`apply_unitaries(...)`](./cirq/protocols/apply_unitaries.md): Apply a series of unitaries onto a state tensor.

[`apply_unitary(...)`](./cirq/protocols/apply_unitary.md): High performance left-multiplication of a unitary effect onto a tensor.

[`approx_eq(...)`](./cirq/protocols/approx_eq.md): Approximately compares two objects.

[`asymmetric_depolarize(...)`](./cirq/ops/asymmetric_depolarize.md): Returns a AsymmetricDepolarizingChannel with given parameter.

[`axis_angle(...)`](./cirq/linalg/axis_angle.md): Decomposes a single-qubit unitary into axis, angle, and global phase.

[`bidiagonalize_real_matrix_pair_with_symmetric_products(...)`](./cirq/linalg/bidiagonalize_real_matrix_pair_with_symmetric_products.md): Finds orthogonal matrices that diagonalize both mat1 and mat2.

[`bidiagonalize_unitary_with_special_orthogonals(...)`](./cirq/linalg/bidiagonalize_unitary_with_special_orthogonals.md): Finds orthogonal matrices L, R such that L @ matrix @ R is diagonal.

[`big_endian_bits_to_int(...)`](./cirq/value/big_endian_bits_to_int.md): Returns the big-endian integer specified by the given bits.

[`big_endian_digits_to_int(...)`](./cirq/value/big_endian_digits_to_int.md): Returns the big-endian integer specified by the given digits and base.

[`big_endian_int_to_bits(...)`](./cirq/value/big_endian_int_to_bits.md): Returns the big-endian bits of an integer.

[`big_endian_int_to_digits(...)`](./cirq/value/big_endian_int_to_digits.md): Separates an integer into big-endian digits.

[`bit_flip(...)`](./cirq/ops/bit_flip.md): Construct a BitFlipChannel that flips a qubit state

[`bloch_vector_from_state_vector(...)`](./cirq/qis/bloch_vector_from_state_vector.md): Returns the bloch vector of a qubit.

[`block_diag(...)`](./cirq/linalg/block_diag.md): Concatenates blocks into a block diagonal matrix.

[`canonicalize_half_turns(...)`](./cirq/value/canonicalize_half_turns.md): Wraps the input into the range (-1, +1].

[`channel(...)`](./cirq/protocols/channel.md): Returns a list of matrices describing the channel for the given value.

[`chosen_angle_to_canonical_half_turns(...)`](./cirq/value/chosen_angle_to_canonical_half_turns.md): Returns a canonicalized half_turns based on the given arguments.

[`chosen_angle_to_half_turns(...)`](./cirq/value/chosen_angle_to_half_turns.md): Returns a half_turns value based on the given arguments.

[`circuit_diagram_info(...)`](./cirq/protocols/circuit_diagram_info.md): Requests information on drawing an operation in a circuit diagram.

[`commutes(...)`](./cirq/protocols/commutes.md): Determines whether two values commute.

[`compute_cphase_exponents_for_fsim_decomposition(...)`](./cirq/optimizers/compute_cphase_exponents_for_fsim_decomposition.md): Returns intervals of CZPowGate exponents valid for FSim decomposition.

[`decompose(...)`](./cirq/protocols/decompose.md): Recursively decomposes a value into <a href="./cirq/ops/Operation.md"><code>cirq.Operation</code></a>s meeting a criteria.

[`decompose_cphase_into_two_fsim(...)`](./cirq/optimizers/decompose_cphase_into_two_fsim.md): Decomposes CZPowGate into two FSimGates.

[`decompose_multi_controlled_rotation(...)`](./cirq/optimizers/decompose_multi_controlled_rotation.md): Implements action of multi-controlled unitary gate.

[`decompose_multi_controlled_x(...)`](./cirq/optimizers/decompose_multi_controlled_x.md): Implements action of multi-controlled Pauli X gate.

[`decompose_once(...)`](./cirq/protocols/decompose_once.md): Decomposes a value into operations, if possible.

[`decompose_once_with_qubits(...)`](./cirq/protocols/decompose_once_with_qubits.md): Decomposes a value into operations on the given qubits.

[`decompose_two_qubit_interaction_into_four_fsim_gates_via_b(...)`](./cirq/optimizers/decompose_two_qubit_interaction_into_four_fsim_gates_via_b.md): Decomposes operations into an FSimGate near theta=pi/2, phi=0.

[`deconstruct_single_qubit_matrix_into_angles(...)`](./cirq/linalg/deconstruct_single_qubit_matrix_into_angles.md): Breaks down a 2x2 unitary into more useful ZYZ angle parameters.

[`definitely_commutes(...)`](./cirq/protocols/definitely_commutes.md): Determines whether two values definitely commute.

[`density_matrix_from_state_vector(...)`](./cirq/qis/density_matrix_from_state_vector.md): Returns the density matrix of the state vector.

[`depolarize(...)`](./cirq/ops/depolarize.md): Returns a DepolarizingChannel with given probability of error.

[`diagonalize_real_symmetric_and_sorted_diagonal_matrices(...)`](./cirq/linalg/diagonalize_real_symmetric_and_sorted_diagonal_matrices.md): Returns an orthogonal matrix that diagonalizes both given matrices.

[`diagonalize_real_symmetric_matrix(...)`](./cirq/linalg/diagonalize_real_symmetric_matrix.md): Returns an orthogonal matrix that diagonalizes the given matrix.

[`dirac_notation(...)`](./cirq/qis/dirac_notation.md): Returns the state vector as a string in Dirac notation.

[`dot(...)`](./cirq/linalg/dot.md): Computes the dot/matrix product of a sequence of values.

[`equal_up_to_global_phase(...)`](./cirq/protocols/equal_up_to_global_phase.md): Determine whether two objects are equal up to global phase.

[`estimate_single_qubit_readout_errors(...)`](./cirq/experiments/estimate_single_qubit_readout_errors.md): Estimate single-qubit readout error.

[`expand_matrix_in_orthogonal_basis(...)`](./cirq/linalg/expand_matrix_in_orthogonal_basis.md): Computes coefficients of expansion of m in basis.

[`eye_tensor(...)`](./cirq/qis/eye_tensor.md): Returns an identity matrix reshaped into a tensor.

[`fidelity(...)`](./cirq/qis/fidelity.md): Fidelity of two quantum states.

[`final_density_matrix(...)`](./cirq/sim/final_density_matrix.md): Returns the density matrix resulting from simulating the circuit.

[`final_state_vector(...)`](./cirq/sim/final_state_vector.md): Returns the state vector resulting from acting operations on a state.

[`final_wavefunction(...)`](./cirq/sim/final_wavefunction.md): THIS FUNCTION IS DEPRECATED.

[`flatten(...)`](./cirq/study/flatten.md): Creates a copy of `val` with any symbols or expressions replaced with

[`flatten_op_tree(...)`](./cirq/ops/flatten_op_tree.md): Performs an in-order iteration of the operations (leaves) in an OP_TREE.

[`flatten_to_ops(...)`](./cirq/ops/flatten_to_ops.md): Performs an in-order iteration of the operations (leaves) in an OP_TREE.

[`flatten_to_ops_or_moments(...)`](./cirq/ops/flatten_to_ops_or_moments.md): Performs an in-order iteration OP_TREE, yielding ops and moments.

[`flatten_with_params(...)`](./cirq/study/flatten_with_params.md): Creates a copy of `val` with any symbols or expressions replaced with

[`flatten_with_sweep(...)`](./cirq/study/flatten_with_sweep.md): Creates a copy of `val` with any symbols or expressions replaced with

[`freeze_op_tree(...)`](./cirq/ops/freeze_op_tree.md): Replaces all iterables in the OP_TREE with tuples.

[`generalized_amplitude_damp(...)`](./cirq/ops/generalized_amplitude_damp.md): Returns a GeneralizedAmplitudeDampingChannel with the given

[`generate_boixo_2018_supremacy_circuits_v2(...)`](./cirq/experiments/generate_boixo_2018_supremacy_circuits_v2.md): Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.

[`generate_boixo_2018_supremacy_circuits_v2_bristlecone(...)`](./cirq/experiments/generate_boixo_2018_supremacy_circuits_v2_bristlecone.md): Generates Google Random Circuits v2 in Bristlecone.

[`generate_boixo_2018_supremacy_circuits_v2_grid(...)`](./cirq/experiments/generate_boixo_2018_supremacy_circuits_v2_grid.md): Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.

[`givens(...)`](./cirq/ops/givens.md): Returns gate with matrix exp(-i angle_rads (Y⊗X - X⊗Y) / 2).

[`has_channel(...)`](./cirq/protocols/has_channel.md): Returns whether the value has a channel representation.

[`has_mixture(...)`](./cirq/protocols/has_mixture.md): Returns whether the value has a mixture representation.

[`has_mixture_channel(...)`](./cirq/protocols/has_mixture_channel.md): THIS FUNCTION IS DEPRECATED.

[`has_stabilizer_effect(...)`](./cirq/protocols/has_stabilizer_effect.md): Returns whether the input has a stabilizer effect.

[`has_unitary(...)`](./cirq/protocols/has_unitary.md): Determines whether the value has a unitary effect.

[`hilbert_schmidt_inner_product(...)`](./cirq/linalg/hilbert_schmidt_inner_product.md): Computes Hilbert-Schmidt inner product of two matrices.

[`hog_score_xeb_fidelity_from_probabilities(...)`](./cirq/experiments/hog_score_xeb_fidelity_from_probabilities.md): XEB fidelity estimator based on normalized HOG score.

[`identity_each(...)`](./cirq/ops/identity_each.md): Returns a single IdentityGate applied to all the given qubits.

[`inverse(...)`](./cirq/protocols/inverse.md): Returns the inverse `val**-1` of the given value, if defined.

[`is_diagonal(...)`](./cirq/linalg/is_diagonal.md): Determines if a matrix is a approximately diagonal.

[`is_hermitian(...)`](./cirq/linalg/is_hermitian.md): Determines if a matrix is approximately Hermitian.

[`is_measurement(...)`](./cirq/protocols/is_measurement.md): Determines whether or not the given value is a measurement.

[`is_native_neutral_atom_gate(...)`](./cirq/neutral_atoms/is_native_neutral_atom_gate.md)

[`is_native_neutral_atom_op(...)`](./cirq/neutral_atoms/is_native_neutral_atom_op.md)

[`is_negligible_turn(...)`](./cirq/optimizers/is_negligible_turn.md)

[`is_normal(...)`](./cirq/linalg/is_normal.md): Determines if a matrix is approximately normal.

[`is_orthogonal(...)`](./cirq/linalg/is_orthogonal.md): Determines if a matrix is approximately orthogonal.

[`is_parameterized(...)`](./cirq/protocols/is_parameterized.md): Returns whether the object is parameterized with any Symbols.

[`is_special_orthogonal(...)`](./cirq/linalg/is_special_orthogonal.md): Determines if a matrix is approximately special orthogonal.

[`is_special_unitary(...)`](./cirq/linalg/is_special_unitary.md): Determines if a matrix is approximately unitary with unit determinant.

[`is_unitary(...)`](./cirq/linalg/is_unitary.md): Determines if a matrix is approximately unitary.

[`json_serializable_dataclass(...)`](./cirq/protocols/json_serializable_dataclass.md): Create a dataclass that supports JSON serialization

[`kak_canonicalize_vector(...)`](./cirq/linalg/kak_canonicalize_vector.md): Canonicalizes an XX/YY/ZZ interaction by swap/negate/shift-ing axes.

[`kak_decomposition(...)`](./cirq/linalg/kak_decomposition.md): Decomposes a 2-qubit unitary into 1-qubit ops and XX/YY/ZZ interactions.

[`kak_vector(...)`](./cirq/linalg/kak_vector.md): Compute the KAK vectors of one or more two qubit unitaries.

[`kron(...)`](./cirq/linalg/kron.md): Computes the kronecker product of a sequence of values.

[`kron_bases(...)`](./cirq/linalg/kron_bases.md): Creates tensor product of bases.

[`kron_factor_4x4_to_2x2s(...)`](./cirq/linalg/kron_factor_4x4_to_2x2s.md): Splits a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

[`kron_with_controls(...)`](./cirq/linalg/kron_with_controls.md): Computes the kronecker product of a sequence of values and control tags.

[`linear_xeb_fidelity(...)`](./cirq/experiments/linear_xeb_fidelity.md): Estimates XEB fidelity from one circuit using linear estimator.

[`linear_xeb_fidelity_from_probabilities(...)`](./cirq/experiments/linear_xeb_fidelity_from_probabilities.md): Linear XEB fidelity estimator.

[`log_xeb_fidelity(...)`](./cirq/experiments/log_xeb_fidelity.md): Estimates XEB fidelity from one circuit using logarithmic estimator.

[`log_xeb_fidelity_from_probabilities(...)`](./cirq/experiments/log_xeb_fidelity_from_probabilities.md): Logarithmic XEB fidelity estimator.

[`map_eigenvalues(...)`](./cirq/linalg/map_eigenvalues.md): Applies a function to the eigenvalues of a matrix.

[`match_global_phase(...)`](./cirq/linalg/match_global_phase.md): Phases the given matrices so that they agree on the phase of one entry.

[`matrix_commutes(...)`](./cirq/linalg/matrix_commutes.md): Determines if two matrices approximately commute.

[`matrix_from_basis_coefficients(...)`](./cirq/linalg/matrix_from_basis_coefficients.md): Computes linear combination of basis vectors with given coefficients.

[`measure(...)`](./cirq/ops/measure.md): Returns a single MeasurementGate applied to all the given qubits.

[`measure_density_matrix(...)`](./cirq/sim/measure_density_matrix.md): Performs a measurement of the density matrix in the computational basis.

[`measure_each(...)`](./cirq/ops/measure_each.md): Returns a list of operations individually measuring the given qubits.

[`measure_state_vector(...)`](./cirq/sim/measure_state_vector.md): Performs a measurement of the state in the computational basis.

[`measurement_key(...)`](./cirq/protocols/measurement_key.md): Get the single measurement key for the given value.

[`measurement_keys(...)`](./cirq/protocols/measurement_keys.md): Gets the measurement keys of measurements within the given value.

[`merge_single_qubit_gates_into_phased_x_z(...)`](./cirq/optimizers/merge_single_qubit_gates_into_phased_x_z.md): Canonicalizes runs of single-qubit rotations in a circuit.

[`merge_single_qubit_gates_into_phxz(...)`](./cirq/optimizers/merge_single_qubit_gates_into_phxz.md): Canonicalizes runs of single-qubit rotations in a circuit.

[`mixture(...)`](./cirq/protocols/mixture.md): Return a sequence of tuples representing a probabilistic unitary.

[`mixture_channel(...)`](./cirq/protocols/mixture_channel.md): THIS FUNCTION IS DEPRECATED.

[`ms(...)`](./cirq/ion/ms.md): Args:

[`mul(...)`](./cirq/protocols/mul.md): Returns lhs * rhs, or else a default if the operator is not implemented.

[`num_qubits(...)`](./cirq/protocols/num_qubits.md): Returns the number of qubits, qudits, or qids `val` operates on.

[`obj_to_dict_helper(...)`](./cirq/protocols/obj_to_dict_helper.md): Construct a dictionary containing attributes from obj

[`one_hot(...)`](./cirq/qis/one_hot.md): Returns a numpy array with all 0s and a single non-zero entry(default 1).

[`partial_trace(...)`](./cirq/linalg/partial_trace.md): Takes the partial trace of a given tensor.

[`partial_trace_of_state_vector_as_mixture(...)`](./cirq/linalg/partial_trace_of_state_vector_as_mixture.md): Returns a mixture representing a state vector with only some qubits kept.

[`pauli_expansion(...)`](./cirq/protocols/pauli_expansion.md): Returns coefficients of the expansion of val in the Pauli basis.

[`phase_by(...)`](./cirq/protocols/phase_by.md): Returns a phased version of the effect.

[`phase_damp(...)`](./cirq/ops/phase_damp.md): Creates a PhaseDampingChannel with damping constant gamma.

[`phase_flip(...)`](./cirq/ops/phase_flip.md): Returns a PhaseFlipChannel that flips a qubit's phase with probability p

[`plot_state_histogram(...)`](./cirq/study/plot_state_histogram.md): Plot the state histogram from a single result with repetitions.

[`pow(...)`](./cirq/protocols/pow.md): Returns `val**factor` of the given value, if defined.

[`pow_pauli_combination(...)`](./cirq/linalg/pow_pauli_combination.md): Computes non-negative integer power of single-qubit Pauli combination.

[`qasm(...)`](./cirq/protocols/qasm.md): Returns QASM code for the given value, if possible.

[`qft(...)`](./cirq/ops/qft.md): The quantum Fourier transform.

[`qid_shape(...)`](./cirq/protocols/qid_shape.md): Returns a tuple describing the number of quantum levels of each

[`quil(...)`](./cirq/protocols/quil.md): Returns the QUIL code for the given value.

[`quirk_json_to_circuit(...)`](./cirq/interop/quirk_json_to_circuit.md): Constructs a Cirq circuit from Quirk's JSON format.

[`quirk_url_to_circuit(...)`](./cirq/interop/quirk_url_to_circuit.md): Parses a Cirq circuit out of a Quirk URL.

[`read_json(...)`](./cirq/protocols/read_json.md): Read a JSON file that optionally contains cirq objects.

[`reflection_matrix_pow(...)`](./cirq/linalg/reflection_matrix_pow.md): Raises a matrix with two opposing eigenvalues to a power.

[`reset(...)`](./cirq/ops/reset.md): Returns a `ResetChannel` on the given qubit.

[`resolve_parameters(...)`](./cirq/protocols/resolve_parameters.md): Resolves symbol parameters in the effect using the param resolver.

[`riswap(...)`](./cirq/ops/riswap.md): Returns gate with matrix exp(+i angle_rads (X⊗X + Y⊗Y) / 2).

[`rx(...)`](./cirq/ops/rx.md): Returns a gate with the matrix e^{-i X rads / 2}.

[`ry(...)`](./cirq/ops/ry.md): Returns a gate with the matrix e^{-i Y rads / 2}.

[`rz(...)`](./cirq/ops/rz.md): Returns a gate with the matrix e^{-i Z rads / 2}.

[`sample(...)`](./cirq/sim/sample.md): Simulates sampling from the given circuit.

[`sample_density_matrix(...)`](./cirq/sim/sample_density_matrix.md): Samples repeatedly from measurements in the computational basis.

[`sample_state_vector(...)`](./cirq/sim/sample_state_vector.md): Samples repeatedly from measurements in the computational basis.

[`sample_sweep(...)`](./cirq/sim/sample_sweep.md): Runs the supplied Circuit, mimicking quantum hardware.

[`scatter_plot_normalized_kak_interaction_coefficients(...)`](./cirq/linalg/scatter_plot_normalized_kak_interaction_coefficients.md): Plots the interaction coefficients of many two-qubit operations.

[`single_qubit_matrix_to_gates(...)`](./cirq/optimizers/single_qubit_matrix_to_gates.md): Implements a single-qubit operation with few gates.

[`single_qubit_matrix_to_pauli_rotations(...)`](./cirq/optimizers/single_qubit_matrix_to_pauli_rotations.md): Implements a single-qubit operation with few rotations.

[`single_qubit_matrix_to_phased_x_z(...)`](./cirq/optimizers/single_qubit_matrix_to_phased_x_z.md): Implements a single-qubit operation with a PhasedX and Z gate.

[`single_qubit_matrix_to_phxz(...)`](./cirq/optimizers/single_qubit_matrix_to_phxz.md): Implements a single-qubit operation with a PhasedXZ gate.

[`single_qubit_op_to_framed_phase_form(...)`](./cirq/optimizers/single_qubit_op_to_framed_phase_form.md): Decomposes a 2x2 unitary M into U^-1 * diag(1, r) * U * diag(g, g).

[`slice_for_qubits_equal_to(...)`](./cirq/linalg/slice_for_qubits_equal_to.md): Returns an index corresponding to a desired subset of an np.ndarray.

[`so4_to_magic_su2s(...)`](./cirq/linalg/so4_to_magic_su2s.md): Finds 2x2 special-unitaries A, B where mat = Mag.H @ kron(A, B) @ Mag.

[`stratified_circuit(...)`](./cirq/optimizers/stratified_circuit.md): Repacks avoiding simultaneous operations with different classes.

[`sub_state_vector(...)`](./cirq/linalg/sub_state_vector.md): Attempts to factor a state vector into two parts and return one of them.

[`subwavefunction(...)`](./cirq/linalg/subwavefunction.md): THIS FUNCTION IS DEPRECATED.

[`targeted_conjugate_about(...)`](./cirq/linalg/targeted_conjugate_about.md): Conjugates the given tensor about the target tensor.

[`targeted_left_multiply(...)`](./cirq/linalg/targeted_left_multiply.md): Left-multiplies the given axes of the target tensor by the given matrix.

[`to_json(...)`](./cirq/protocols/to_json.md): Write a JSON file containing a representation of obj.

[`to_resolvers(...)`](./cirq/study/to_resolvers.md): Convert a Sweepable to a list of ParamResolvers.

[`to_sweep(...)`](./cirq/study/to_sweep.md): Converts the argument into a `<a href="./cirq/study/Sweep.md"><code>cirq.Sweep</code></a>`.

[`to_sweeps(...)`](./cirq/study/to_sweeps.md): Converts a Sweepable to a list of Sweeps.

[`to_valid_density_matrix(...)`](./cirq/qis/to_valid_density_matrix.md): Verifies the density_matrix_rep is valid and converts it to ndarray form.

[`to_valid_state_vector(...)`](./cirq/qis/to_valid_state_vector.md): Verifies the state_rep is valid and converts it to ndarray form.

[`trace_distance_bound(...)`](./cirq/protocols/trace_distance_bound.md): Returns a maximum on the trace distance between this effect's input

[`trace_distance_from_angle_list(...)`](./cirq/protocols/trace_distance_from_angle_list.md): Given a list of arguments of the eigenvalues of a unitary matrix,

[`transform_op_tree(...)`](./cirq/ops/transform_op_tree.md): Maps transformation functions onto the nodes of an OP_TREE.

[`two_qubit_matrix_to_ion_operations(...)`](./cirq/ion/two_qubit_matrix_to_ion_operations.md): Decomposes a two-qubit operation into MS/single-qubit rotation gates.

[`two_qubit_matrix_to_operations(...)`](./cirq/optimizers/two_qubit_matrix_to_operations.md): Decomposes a two-qubit operation into Z/XY/CZ gates.

[`unitary(...)`](./cirq/protocols/unitary.md): Returns a unitary matrix describing the given value.

[`unitary_eig(...)`](./cirq/linalg/unitary_eig.md): Gives the guaranteed unitary eigendecomposition of a normal matrix.

[`validate_indices(...)`](./cirq/qis/validate_indices.md): Validates that the indices have values within range of num_qubits.

[`validate_mixture(...)`](./cirq/protocols/validate_mixture.md): Validates that the mixture's tuple are valid probabilities.

[`validate_normalized_state(...)`](./cirq/qis/validate_normalized_state.md): THIS FUNCTION IS DEPRECATED.

[`validate_normalized_state_vector(...)`](./cirq/qis/validate_normalized_state_vector.md): Validates that the given state vector is a valid.

[`validate_probability(...)`](./cirq/value/validate_probability.md): Validates that a probability is between 0 and 1 inclusively.

[`validate_qid_shape(...)`](./cirq/qis/validate_qid_shape.md): Validates the size of the given `state_vector` against the given shape.

[`value_equality(...)`](./cirq/value/value_equality.md): Implements __eq__/__ne__/__hash__ via a _value_equality_values_ method.

[`von_neumann_entropy(...)`](./cirq/qis/von_neumann_entropy.md): Calculates von Neumann entropy of density matrix in bits.

[`wavefunction_partial_trace_as_mixture(...)`](./cirq/linalg/wavefunction_partial_trace_as_mixture.md): THIS FUNCTION IS DEPRECATED.

[`xeb_fidelity(...)`](./cirq/experiments/xeb_fidelity.md): Estimates XEB fidelity from one circuit using user-supplied estimator.

## Type Aliases

[`CIRCUIT_LIKE`](./cirq/sim/CIRCUIT_LIKE.md)

[`DURATION_LIKE`](./cirq/value/DURATION_LIKE.md)

[`NOISE_MODEL_LIKE`](./cirq/devices/NOISE_MODEL_LIKE.md)

[`OP_TREE`](./cirq/ops/OP_TREE.md)

[`PAULI_GATE_LIKE`](./cirq/ops/PAULI_GATE_LIKE.md)

[`PAULI_STRING_LIKE`](./cirq/ops/PAULI_STRING_LIKE.md)

[`ParamDictType`](./cirq/study/ParamDictType.md)

[`ParamResolverOrSimilarType`](./cirq/study/ParamResolverOrSimilarType.md)

[`PauliSumLike`](./cirq/ops/PauliSumLike.md)

[`QubitOrderOrList`](./cirq/ops/QubitOrderOrList.md)

[`STATE_VECTOR_LIKE`](./cirq/qis/STATE_VECTOR_LIKE.md)

[`Sweepable`](./cirq/study/Sweepable.md)

[`TParamVal`](./cirq/value/TParamVal.md)

## Other Members

* `CONTROL_TAG` <a id="CONTROL_TAG"></a>
* `DEFAULT_RESOLVERS` <a id="DEFAULT_RESOLVERS"></a>
* `NO_NOISE` <a id="NO_NOISE"></a>
* `PAULI_BASIS` <a id="PAULI_BASIS"></a>
* `UNCONSTRAINED_DEVICE` <a id="UNCONSTRAINED_DEVICE"></a>
* `UnitSweep` <a id="UnitSweep"></a>
* `__version__ = '0.9.0.dev'` <a id="__version__"></a>
