<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.merge_single_qubit_gates_into_phxz" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.merge_single_qubit_gates_into_phxz

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/merge_single_qubit_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Canonicalizes runs of single-qubit rotations in a circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.merge_single_qubit_gates_into_phxz`, `cirq.optimizers.merge_single_qubit_gates.merge_single_qubit_gates_into_phxz`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.merge_single_qubit_gates_into_phxz(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    atol: float = 1e-08
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Specifically, any run of non-parameterized single-qubit gates will be
replaced by an optional PhasedXZ operation.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit to rewrite. This value is mutated in-place.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute tolerance to angle error. Larger values allow more
negligible gates to be dropped, smaller values increase accuracy.
</td>
</tr>
</table>

