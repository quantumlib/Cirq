<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.targeted_left_multiply" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.targeted_left_multiply

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/transformations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Left-multiplies the given axes of the target tensor by the given matrix.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.transformations.targeted_left_multiply`, `cirq.targeted_left_multiply`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.targeted_left_multiply(
    left_matrix: np.ndarray,
    right_target: np.ndarray,
    target_axes: Sequence[int],
    out: Optional[np.ndarray] = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Note that the matrix must have a compatible tensor structure.

For example, if you have an 6-qubit state vector `input_state` with shape
(2, 2, 2, 2, 2, 2), and a 2-qubit unitary operation `op` with shape
(2, 2, 2, 2), and you want to apply `op` to the 5'th and 3'rd qubits
within `input_state`, then the output state vector is computed as follows:

    output_state = cirq.targeted_left_multiply(op, input_state, [5, 3])

This method also works when the right hand side is a matrix instead of a
vector. If a unitary circuit's matrix is `old_effect`, and you append
a CNOT(q1, q4) operation onto the circuit, where the control q1 is the qubit
at offset 1 and the target q4 is the qubit at offset 4, then the appended
circuit's unitary matrix is computed as follows:

    new_effect = cirq.targeted_left_multiply(
        left_matrix=cirq.unitary(cirq.CNOT).reshape((2, 2, 2, 2)),
        right_target=old_effect,
        target_axes=[1, 4])

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`left_matrix`
</td>
<td>
What to left-multiply the target tensor by.
</td>
</tr><tr>
<td>
`right_target`
</td>
<td>
A tensor to carefully broadcast a left-multiply over.
</td>
</tr><tr>
<td>
`target_axes`
</td>
<td>
Which axes of the target are being operated on.
</td>
</tr><tr>
<td>
`out`
</td>
<td>
The buffer to store the results in. If not specified or None, a new
buffer is used. Must have the same shape as right_target.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The output tensor.
</td>
</tr>

</table>

