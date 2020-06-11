<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="GRID_ALIGNED_PATTERN"/>
<meta itemprop="property" content="GRID_STAGGERED_PATTERN"/>
</div>

# Module: cirq.experiments

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Modules

[`fidelity_estimation`](../cirq/experiments/fidelity_estimation.md) module: Estimation of fidelity associated with experimental circuit executions.

[`google_v2_supremacy_circuit`](../cirq/experiments/google_v2_supremacy_circuit.md) module

[`n_qubit_tomography`](../cirq/experiments/n_qubit_tomography.md) module: Tomography code for an arbitrary number of qubits allowing for

[`qubit_characterizations`](../cirq/experiments/qubit_characterizations.md) module

[`random_quantum_circuit_generation`](../cirq/experiments/random_quantum_circuit_generation.md) module: Code for generating random quantum circuits.

[`single_qubit_readout_calibration`](../cirq/experiments/single_qubit_readout_calibration.md) module

[`t1_decay_experiment`](../cirq/experiments/t1_decay_experiment.md) module

[`t2_decay_experiment`](../cirq/experiments/t2_decay_experiment.md) module

## Classes

[`class CrossEntropyResult`](../cirq/experiments/CrossEntropyResult.md): Results from a cross-entropy benchmarking (XEB) experiment.

[`class GridInteractionLayer`](../cirq/experiments/GridInteractionLayer.md): A layer of aligned or staggered two-qubit interactions on a grid.

[`class RabiResult`](../cirq/experiments/RabiResult.md): Results from a Rabi oscillation experiment.

[`class RandomizedBenchMarkResult`](../cirq/experiments/RandomizedBenchMarkResult.md): Results from a randomized benchmarking experiment.

[`class SingleQubitReadoutCalibrationResult`](../cirq/experiments/SingleQubitReadoutCalibrationResult.md): Result of estimating single qubit readout error.

[`class StateTomographyExperiment`](../cirq/experiments/StateTomographyExperiment.md): Experiment to conduct state tomography.

[`class T1DecayResult`](../cirq/experiments/T1DecayResult.md): Results from a Rabi oscillation experiment.

[`class T2DecayResult`](../cirq/experiments/T2DecayResult.md): Results from a T2 decay experiment.

[`class TomographyResult`](../cirq/experiments/TomographyResult.md): Results from a state tomography experiment.

## Functions

[`build_entangling_layers(...)`](../cirq/experiments/build_entangling_layers.md): Builds a sequence of gates that entangle all pairs of qubits on a grid.

[`cross_entropy_benchmarking(...)`](../cirq/experiments/cross_entropy_benchmarking.md): Cross-entropy benchmarking (XEB) of multiple qubits.

[`estimate_single_qubit_readout_errors(...)`](../cirq/experiments/estimate_single_qubit_readout_errors.md): Estimate single-qubit readout error.

[`generate_boixo_2018_supremacy_circuits_v2(...)`](../cirq/experiments/generate_boixo_2018_supremacy_circuits_v2.md): Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.

[`generate_boixo_2018_supremacy_circuits_v2_bristlecone(...)`](../cirq/experiments/generate_boixo_2018_supremacy_circuits_v2_bristlecone.md): Generates Google Random Circuits v2 in Bristlecone.

[`generate_boixo_2018_supremacy_circuits_v2_grid(...)`](../cirq/experiments/generate_boixo_2018_supremacy_circuits_v2_grid.md): Generates Google Random Circuits v2 as in github.com/sboixo/GRCS cz_v2.

[`get_state_tomography_data(...)`](../cirq/experiments/get_state_tomography_data.md): Gets the data for each rotation string added to the circuit.

[`hog_score_xeb_fidelity_from_probabilities(...)`](../cirq/experiments/hog_score_xeb_fidelity_from_probabilities.md): XEB fidelity estimator based on normalized HOG score.

[`linear_xeb_fidelity(...)`](../cirq/experiments/linear_xeb_fidelity.md): Estimates XEB fidelity from one circuit using linear estimator.

[`linear_xeb_fidelity_from_probabilities(...)`](../cirq/experiments/linear_xeb_fidelity_from_probabilities.md): Linear XEB fidelity estimator.

[`log_xeb_fidelity(...)`](../cirq/experiments/log_xeb_fidelity.md): Estimates XEB fidelity from one circuit using logarithmic estimator.

[`log_xeb_fidelity_from_probabilities(...)`](../cirq/experiments/log_xeb_fidelity_from_probabilities.md): Logarithmic XEB fidelity estimator.

[`rabi_oscillations(...)`](../cirq/experiments/rabi_oscillations.md): Runs a Rabi oscillation experiment.

[`random_rotations_between_grid_interaction_layers_circuit(...)`](../cirq/experiments/random_rotations_between_grid_interaction_layers_circuit.md): Generate a random quantum circuit.

[`single_qubit_randomized_benchmarking(...)`](../cirq/experiments/single_qubit_randomized_benchmarking.md): Clifford-based randomized benchmarking (RB) of a single qubit.

[`single_qubit_state_tomography(...)`](../cirq/experiments/single_qubit_state_tomography.md): Single-qubit state tomography.

[`state_tomography(...)`](../cirq/experiments/state_tomography.md): This performs n qubit tomography on a cirq circuit

[`t1_decay(...)`](../cirq/experiments/t1_decay.md): Runs a t1 decay experiment.

[`t2_decay(...)`](../cirq/experiments/t2_decay.md): Runs a t2 transverse relaxation experiment.

[`two_qubit_randomized_benchmarking(...)`](../cirq/experiments/two_qubit_randomized_benchmarking.md): Clifford-based randomized benchmarking (RB) of two qubits.

[`two_qubit_state_tomography(...)`](../cirq/experiments/two_qubit_state_tomography.md): Two-qubit state tomography.

[`xeb_fidelity(...)`](../cirq/experiments/xeb_fidelity.md): Estimates XEB fidelity from one circuit using user-supplied estimator.

## Other Members

* `GRID_ALIGNED_PATTERN` <a id="GRID_ALIGNED_PATTERN"></a>
* `GRID_STAGGERED_PATTERN` <a id="GRID_STAGGERED_PATTERN"></a>
