<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.n_qubit_tomography" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.experiments.n_qubit_tomography

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/n_qubit_tomography.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tomography code for an arbitrary number of qubits allowing for

different pre-measurement rotations.

The code is designed to be modular with regards to data collection
so that occurs outside of the StateTomographyExperiment class.

## Classes

[`class StateTomographyExperiment`](../../cirq/experiments/StateTomographyExperiment.md): Experiment to conduct state tomography.

## Functions

[`get_state_tomography_data(...)`](../../cirq/experiments/get_state_tomography_data.md): Gets the data for each rotation string added to the circuit.

[`state_tomography(...)`](../../cirq/experiments/state_tomography.md): This performs n qubit tomography on a cirq circuit

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
