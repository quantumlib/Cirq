<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.decompose_multi_controlled_rotation" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.decompose_multi_controlled_rotation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/controlled_gate_decomposition.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implements action of multi-controlled unitary gate.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.decompose_multi_controlled_rotation`, `cirq.optimizers.controlled_gate_decomposition.decompose_multi_controlled_rotation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.decompose_multi_controlled_rotation(
    matrix: np.ndarray,
    controls: List['cirq.Qid'],
    target: "cirq.Qid"
) -> List['cirq.Operation']
</code></pre>



<!-- Placeholder for "Used in" -->

Returns a sequence of operations, which is equivalent to applying
single-qubit gate with matrix `matrix` on `target`, controlled by
`controls`.

Result is guaranteed to consist exclusively of 1-qubit, CNOT and CCNOT
gates.

If matrix is special unitary, result has length `O(len(controls))`.
Otherwise result has length `O(len(controls)**2)`.

#### References:

[1] Barenco, Bennett et al.
    Elementary gates for quantum computation. 1995.
    https://arxiv.org/pdf/quant-ph/9503016.pdf



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>
<tr class="alt">
<td colspan="2">
matrix - 2x2 numpy unitary matrix (of real or complex dtype).
controls - control qubits.
targets - target qubits.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of operations which, applied in a sequence, are equivalent to
applying `MatrixGate(matrix).on(target).controlled_by(*controls)`.
</td>
</tr>

</table>

