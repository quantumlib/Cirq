<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.state_tomography" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.state_tomography

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/n_qubit_tomography.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



This performs n qubit tomography on a cirq circuit

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.n_qubit_tomography.state_tomography`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.state_tomography(
    sampler: "cirq.Sampler",
    qubits: Sequence['cirq.Qid'],
    circuit: "cirq.Circuit",
    repetitions: int = 1000,
    prerotations: Sequence[Tuple[float, float]] = None
) -> <a href="../../cirq/experiments/TomographyResult.md"><code>cirq.experiments.TomographyResult</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Follows https://web.physics.ucsb.edu/~martinisgroup/theses/Neeley2010b.pdf
A.1. State Tomography.
This is a high level interface for StateTomographyExperiment.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
Circuit to do the tomography on.
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
`sampler`
</td>
<td>
Sampler to collect the data from.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
Number of times to sample each rotation.
</td>
</tr><tr>
<td>
`prerotations`
</td>
<td>
Tuples of (phase_exponent, exponent) parameters for gates
to apply to the qubits before measurement. The actual rotation
applied will be <a href="../../cirq/ops/PhasedXPowGate.md"><code>cirq.PhasedXPowGate</code></a> with the specified values
of phase_exponent and exponent. If None, we use [(0, 0), (0, 0.5),
(0.5, 0.5)], which corresponds to rotation gates
[I, X**0.5, Y**0.5].
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
`TomographyResult` which contains the density matrix of the qubits
determined by tomography.
</td>
</tr>

</table>

