<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.sub_state_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.sub_state_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/transformations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Attempts to factor a state vector into two parts and return one of them.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.transformations.sub_state_vector`, `cirq.sub_state_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.sub_state_vector(
    state_vector: np.ndarray,
    keep_indices: List[int],
    *,
    default: <a href="../../cirq/linalg/transformations/TDefault.md"><code>cirq.linalg.transformations.TDefault</code></a> = cirq.linalg.transformations.RaiseValueErrorIfNotProvided,
    atol: Union[int, float] = 1e-08
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

The input `state_vector` must have shape ``(2,) * n`` or ``(2 ** n)`` where
`state_vector` is expressed over n qubits. The returned array will retain
the same type of shape as the input state vector, either ``(2 ** k)`` or
``(2,) * k`` where k is the number of qubits kept.

If a state vector $|\psi\rangle$ defined on n qubits is an outer product
of kets like  $|\psi\rangle$ = $|x\rangle \otimes |y\rangle$, and
$|x\rangle$ is defined over the subset ``keep_indices`` of k qubits, then
this method will factor $|\psi\rangle$ into $|x\rangle$ and $|y\rangle$ and
return $|x\rangle$. Note that $|x\rangle$ is not unique, because scalar
multiplication may be absorbed by any factor of a tensor product,
$e^{i \theta} |y\rangle \otimes |x\rangle =
|y\rangle \otimes e^{i \theta} |x\rangle$

This method randomizes the global phase of $|x\rangle$ in order to avoid
accidental reliance on the global phase being some specific value.

If the provided `state_vector` cannot be factored into a pure state over
`keep_indices`, the method will fall back to return `default`. If `default`
is not provided, the method will fail and raise `ValueError`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state_vector`
</td>
<td>
The target state_vector.
</td>
</tr><tr>
<td>
`keep_indices`
</td>
<td>
Which indices to attempt to get the separable part of the
`state_vector` on.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Determines the fallback behavior when `state_vector` doesn't
have a pure state factorization. If the factored state is not pure
and `default` is not set, a ValueError is raised. If default is set
to a value, that value is returned.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
The minimum tolerance for comparing the output state's coherence
measure to 1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The state vector expressed over the desired subset of qubits.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the `state_vector` is not of the correct shape or the
indices are not a valid subset of the input `state_vector`'s indices, or
the result of factoring is not a pure state.
</td>
</tr>
</table>

