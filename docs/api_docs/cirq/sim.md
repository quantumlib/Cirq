<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.sim" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="deprecated_constants"/>
</div>

# Module: cirq.sim

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/sim/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base simulation classes and generic simulators.



## Modules

[`act_on_state_vector_args`](../cirq/sim/act_on_state_vector_args.md) module: Objects and methods for acting efficiently on a state vector.

[`clifford`](../cirq/sim/clifford.md) module

[`density_matrix_simulator`](../cirq/sim/density_matrix_simulator.md) module: Simulator for density matrices that simulates noisy quantum circuits.

[`density_matrix_utils`](../cirq/sim/density_matrix_utils.md) module: Code to handle density matrices.

[`mux`](../cirq/sim/mux.md) module: Sampling/simulation methods that delegate to appropriate simulators.

[`simulator`](../cirq/sim/simulator.md) module: Abstract base classes for different types of simulators.

[`sparse_simulator`](../cirq/sim/sparse_simulator.md) module: A simulator that uses numpy's einsum for sparse matrix operations.

[`state_vector`](../cirq/sim/state_vector.md) module: Helpers for handling quantum state vectors.

[`state_vector_simulator`](../cirq/sim/state_vector_simulator.md) module: Abstract classes for simulations which keep track of state vector.

## Classes

[`class ActOnStateVectorArgs`](../cirq/sim/ActOnStateVectorArgs.md): State and context for an operation acting on a state vector.

[`class CliffordSimulator`](../cirq/sim/CliffordSimulator.md): An efficient simulator for Clifford circuits.

[`class CliffordSimulatorStepResult`](../cirq/sim/CliffordSimulatorStepResult.md): A `StepResult` that includes `StateVectorMixin` methods.

[`class CliffordState`](../cirq/sim/CliffordState.md): A state of the Clifford simulation.

[`class CliffordTableau`](../cirq/sim/CliffordTableau.md): Tableau representation of a stabilizer state

[`class CliffordTrialResult`](../cirq/sim/CliffordTrialResult.md): Results of a simulation by a SimulatesFinalState.

[`class DensityMatrixSimulator`](../cirq/sim/DensityMatrixSimulator.md): A simulator for density matrices and noisy quantum circuits.

[`class DensityMatrixSimulatorState`](../cirq/sim/DensityMatrixSimulatorState.md): The simulator state for DensityMatrixSimulator

[`class DensityMatrixStepResult`](../cirq/sim/DensityMatrixStepResult.md): A single step in the simulation of the DensityMatrixSimulator.

[`class DensityMatrixTrialResult`](../cirq/sim/DensityMatrixTrialResult.md): A `SimulationTrialResult` for `DensityMatrixSimulator` runs.

[`class SimulatesAmplitudes`](../cirq/sim/SimulatesAmplitudes.md): Simulator that computes final amplitudes of given bitstrings.

[`class SimulatesFinalState`](../cirq/sim/SimulatesFinalState.md): Simulator that allows access to the simulator's final state.

[`class SimulatesIntermediateState`](../cirq/sim/SimulatesIntermediateState.md): A SimulatesFinalState that simulates a circuit by moments.

[`class SimulatesIntermediateStateVector`](../cirq/sim/SimulatesIntermediateStateVector.md): A simulator that accesses its state vector as it does its simulation.

[`class SimulatesIntermediateWaveFunction`](../cirq/sim/SimulatesIntermediateWaveFunction.md): Deprecated. Please use `SimulatesIntermediateStateVector` instead.

[`class SimulatesSamples`](../cirq/sim/SimulatesSamples.md): Simulator that mimics running on quantum hardware.

[`class SimulationTrialResult`](../cirq/sim/SimulationTrialResult.md): Results of a simulation by a SimulatesFinalState.

[`class Simulator`](../cirq/sim/Simulator.md): A sparse matrix state vector simulator that uses numpy.

[`class SparseSimulatorStep`](../cirq/sim/SparseSimulatorStep.md): A `StepResult` that includes `StateVectorMixin` methods.

[`class StabilizerStateChForm`](../cirq/sim/StabilizerStateChForm.md): A representation of stabilizer states using the CH form,

[`class StateVectorMixin`](../cirq/sim/StateVectorMixin.md): A mixin that provide methods for objects that have a state vector.

[`class StateVectorSimulatorState`](../cirq/sim/StateVectorSimulatorState.md)

[`class StateVectorStepResult`](../cirq/sim/StateVectorStepResult.md): Results of a step of a SimulatesIntermediateState.

[`class StateVectorTrialResult`](../cirq/sim/StateVectorTrialResult.md): A `SimulationTrialResult` that includes the `StateVectorMixin` methods.

[`class StepResult`](../cirq/sim/StepResult.md): Results of a step of a SimulatesIntermediateState.

[`class WaveFunctionSimulatorState`](../cirq/sim/WaveFunctionSimulatorState.md): Deprecated. Please use `StateVectorSimulatorState` instead.

[`class WaveFunctionStepResult`](../cirq/sim/WaveFunctionStepResult.md): Deprecated. Please use `StateVectorStepResult` instead.

[`class WaveFunctionTrialResult`](../cirq/sim/WaveFunctionTrialResult.md): Deprecated. Please use `StateVectorTrialResult` instead.

## Functions

[`final_density_matrix(...)`](../cirq/sim/final_density_matrix.md): Returns the density matrix resulting from simulating the circuit.

[`final_state_vector(...)`](../cirq/sim/final_state_vector.md): Returns the state vector resulting from acting operations on a state.

[`final_wavefunction(...)`](../cirq/sim/final_wavefunction.md): THIS FUNCTION IS DEPRECATED.

[`measure_density_matrix(...)`](../cirq/sim/measure_density_matrix.md): Performs a measurement of the density matrix in the computational basis.

[`measure_state_vector(...)`](../cirq/sim/measure_state_vector.md): Performs a measurement of the state in the computational basis.

[`sample(...)`](../cirq/sim/sample.md): Simulates sampling from the given circuit.

[`sample_density_matrix(...)`](../cirq/sim/sample_density_matrix.md): Samples repeatedly from measurements in the computational basis.

[`sample_state_vector(...)`](../cirq/sim/sample_state_vector.md): Samples repeatedly from measurements in the computational basis.

[`sample_sweep(...)`](../cirq/sim/sample_sweep.md): Runs the supplied Circuit, mimicking quantum hardware.

## Type Aliases

[`CIRCUIT_LIKE`](../cirq/sim/CIRCUIT_LIKE.md)

[`STATE_VECTOR_LIKE`](../cirq/qis/STATE_VECTOR_LIKE.md)

## Other Members

* `deprecated_constants` <a id="deprecated_constants"></a>
