<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.decompose_multi_controlled_x" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.decompose_multi_controlled_x

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/controlled_gate_decomposition.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implements action of multi-controlled Pauli X gate.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.decompose_multi_controlled_x`, `cirq.optimizers.controlled_gate_decomposition.decompose_multi_controlled_x`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.decompose_multi_controlled_x(
    controls: List['cirq.Qid'],
    target: "cirq.Qid",
    free_qubits: List['cirq.Qid']
) -> List['cirq.Operation']
</code></pre>



<!-- Placeholder for "Used in" -->

Result is guaranteed to consist exclusively of 1-qubit, CNOT and CCNOT
gates.
If `free_qubits` has at least 1 element, result has lengts
O(len(controls)).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
controls - control qubits.
targets - target qubits.
free_qubits - qubits which are neither controlled nor target. Can be
modified by algorithm, but will end up in their initial state.
</td>
</tr>

</table>

