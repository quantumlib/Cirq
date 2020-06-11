<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.estimate_single_qubit_readout_errors" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.estimate_single_qubit_readout_errors

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/single_qubit_readout_calibration.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Estimate single-qubit readout error.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.estimate_single_qubit_readout_errors`, `cirq.experiments.single_qubit_readout_calibration.estimate_single_qubit_readout_errors`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.estimate_single_qubit_readout_errors(
    sampler: "cirq.Sampler",
    *,
    qubits: Iterable['cirq.Qid'] = 1000,
    repetitions: int = 1000
) -> <a href="../../cirq/experiments/SingleQubitReadoutCalibrationResult.md"><code>cirq.experiments.SingleQubitReadoutCalibrationResult</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

For each qubit, prepare the |0⟩ state and measure. Calculate how often a 1
is measured. Also, prepare the |1⟩ state and calculate how often a 0 is
measured. The state preparations and measurements are done in parallel,
i.e., for the first experiment, we actually prepare every qubit in the |0⟩
state and measure them simultaneously.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sampler`
</td>
<td>
The quantum engine or simulator to run the circuits.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
The qubits being tested.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
The number of measurement repetitions to perform.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A SingleQubitReadoutCalibrationResult storing the readout error
probabilities as well as the number of repetitions used to estimate
the probabilties. Also stores a timestamp indicating the time when
data was finished being collected from the sampler.
</td>
</tr>

</table>

