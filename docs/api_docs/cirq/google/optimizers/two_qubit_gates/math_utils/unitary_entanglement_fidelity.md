<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.two_qubit_gates.math_utils.unitary_entanglement_fidelity" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.two_qubit_gates.math_utils.unitary_entanglement_fidelity

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/math_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Entanglement fidelity between two unitaries.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.two_qubit_gates.math_utils.unitary_entanglement_fidelity(
    U_actual: np.ndarray,
    U_ideal: np.ndarray
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

For unitary matrices, this is related to the average unitary fidelity F
as

:math:`F = \frac{F_e d + 1}{d + 1}`
where d is the matrix dimension.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`U_actual`
</td>
<td>
Matrix whose fidelity to U_ideal will be computed. This may
be a non-unitary matrix, i.e. the projection of a larger unitary
matrix into the computational subspace.
</td>
</tr><tr>
<td>
`U_ideal`
</td>
<td>
Unitary matrix to which U_actual will be compared.
</td>
</tr>
</table>


Both arguments may be vectorized, in that their shapes may be of the form
(...,M,M) (as long as both shapes can be broadcast together).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The entanglement fidelity between the two unitaries. For inputs with
shape (...,M,M), the output has shape (...).
</td>
</tr>

</table>

