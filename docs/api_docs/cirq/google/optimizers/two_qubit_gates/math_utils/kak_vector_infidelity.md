<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.two_qubit_gates.math_utils.kak_vector_infidelity" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.two_qubit_gates.math_utils.kak_vector_infidelity

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/math_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The locally invariant infidelity between two KAK vectors.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.two_qubit_gates.math_utils.kak_vector_infidelity(
    k_vec_a: np.ndarray,
    k_vec_b: np.ndarray,
    ignore_equivalent_vectors: bool = False
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

This is the quantity

\min 1 - F_e( exp(i k_a · (XX,YY,ZZ)) kL exp(i k_b · (XX,YY,ZZ)) kR)

where F_e is the entanglement (process) fidelity and the minimum is taken
over all 1-local unitaries kL, kR.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`k_vec_a`
</td>
<td>
A 3-vector or tensor of 3-vectors with shape (...,3).
</td>
</tr><tr>
<td>
`k_vec_b`
</td>
<td>
A 3-vector or tensor of 3-vectors with shape (...,3). If both
k_vec_a and k_vec_b are tensors, their shapes must be compatible
for broadcasting.
</td>
</tr><tr>
<td>
`ignore_equivalent_vectors`
</td>
<td>
If True, the calculation ignores any other
KAK vectors that are equivalent to the inputs under local unitaries.
The resulting infidelity is then only an upper bound to the true
infidelity.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An ndarray storing the locally invariant infidelity between the inputs.
If k_vec_a or k_vec_b is a tensor, the result is vectorized.
</td>
</tr>

</table>

