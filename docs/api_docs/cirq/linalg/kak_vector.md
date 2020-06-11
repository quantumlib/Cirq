<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.kak_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.kak_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Compute the KAK vectors of one or more two qubit unitaries.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.kak_vector`, `cirq.linalg.decompositions.kak_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.kak_vector(
    unitary: Union[Iterable[np.ndarray], np.ndarray],
    *,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    check_preconditions: bool = True
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

Any 2 qubit unitary may be expressed as

$$ U = k_l A k_r $$
where $k_l, k_r$ are single qubit (local) unitaries and

$$ A= \exp \left(i \sum_{s=x,y,z} k_s \sigma_{s}^{(0)} \sigma_{s}^{(1)}
             \right) $$

The vector entries are ordered such that
    $$ 0 ≤ |k_z| ≤ k_y ≤ k_x ≤ π/4 $$
if $k_x$ = π/4, $k_z \geq 0$.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`unitary`
</td>
<td>
A unitary matrix, or a multi-dimensional array of unitary
matrices. Must have shape (..., 4, 4), where the last two axes are
for the unitary matrix and other axes are for broadcasting the kak
vector computation.
</td>
</tr><tr>
<td>
`rtol`
</td>
<td>
Per-matrix-entry relative tolerance on equality. Used in unitarity
check of input.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Per-matrix-entry absolute tolerance on equality. Used in unitarity
check of input. This also determines how close $k_x$ must be to π/4
to guarantee $k_z$ ≥ 0. Must be non-negative.
</td>
</tr><tr>
<td>
`check_preconditions`
</td>
<td>
When set to False, skips verifying that the input
is unitary in order to increase performance.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The KAK vector of the given unitary or unitaries. The output shape is
the same as the input shape, except the two unitary matrix axes are
replaced by the kak vector axis (i.e. the output has shape
`unitary.shape[:-2] + (3,)`).
</td>
</tr>

</table>



#### References:

The appendix section of "Lower bounds on the complexity of simulating
quantum gates".
http://arxiv.org/abs/quant-ph/0307190v1



#### Examples:

>>> cirq.kak_vector(np.eye(4))
array([0., 0., 0.])
>>> unitaries = [cirq.unitary(cirq.CZ),cirq.unitary(cirq.ISWAP)]
>>> cirq.kak_vector(unitaries) * 4 / np.pi
array([[ 1.,  0., -0.],
       [ 1.,  1.,  0.]])
