<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.get_state_tomography_data" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.get_state_tomography_data

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/n_qubit_tomography.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the data for each rotation string added to the circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.n_qubit_tomography.get_state_tomography_data`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.get_state_tomography_data(
    sampler: "cirq.Sampler",
    qubits: Sequence['cirq.Qid'],
    circuit: "cirq.Circuit",
    rot_circuit: "cirq.Circuit",
    rot_sweep: "cirq.Sweep",
    repetitions: int = 1000
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

For each sequence in prerotation_sequences gets the probability of all
2**n bit strings.  Resulting matrix will have dimensions
(len(rot_sweep)**n, 2**n).
This is a default way to get data that can be replaced by the user if they
have a more advanced protocol in mind.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sampler`
</td>
<td>
Sampler to collect the data from.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
Qubits to do the tomography on.
</td>
</tr><tr>
<td>
`circuit`
</td>
<td>
Circuit to do the tomography on.
</td>
</tr><tr>
<td>
`rot_circuit`
</td>
<td>
Circuit with parameterized rotation gates to do before the
final measurements.
</td>
</tr><tr>
<td>
`rot_sweep`
</td>
<td>
The list of rotations on the qubits to perform before
measurement.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
Number of times to sample each rotation sequence.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
2D array of probabilities, where first index is which pre-rotation was
applied and second index is the qubit state.
</td>
</tr>

</table>

