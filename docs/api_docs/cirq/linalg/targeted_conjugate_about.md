<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.targeted_conjugate_about" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.targeted_conjugate_about

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/transformations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Conjugates the given tensor about the target tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.transformations.targeted_conjugate_about`, `cirq.targeted_conjugate_about`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.targeted_conjugate_about(
    tensor: np.ndarray,
    target: np.ndarray,
    indices: Sequence[int],
    conj_indices: Sequence[int] = None,
    buffer: Optional[np.ndarray] = None,
    out: Optional[np.ndarray] = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

This method computes a target tensor conjugated by another tensor.
Here conjugate is used in the sense of conjugating by a matrix, i.a.
A conjugated about B is $A B A^\dagger$ where $\dagger$ represents the
conjugate transpose.

Abstractly this compute $A \cdot B \cdot A^\dagger$ where A and B are
multi-dimensional arrays, and instead of matrix multiplication $\cdot$
is a contraction between the given indices (indices for first $\cdot$,
conj_indices for second $\cdot$).

More specifically this computes
    sum tensor_{i_0,...,i_{r-1},j_0,...,j_{r-1}}
    * target_{k_0,...,k_{r-1},l_0,...,l_{r-1}
    * tensor_{m_0,...,m_{r-1},n_0,...,n_{r-1}}^*
where the sum is over indices where j_s = k_s and s is in `indices`
and l_s = m_s and s is in `conj_indices`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
The tensor that will be conjugated about the target tensor.
</td>
</tr><tr>
<td>
`target`
</td>
<td>
The tensor that will receive the conjugation.
</td>
</tr><tr>
<td>
`indices`
</td>
<td>
The indices which will be contracted between the tensor and
target.
conj_indices; The indices which will be contracted between the
complex conjugate of the tensor and the target. If this is None,
then these will be the values in indices plus half the number
of dimensions of the target (`ndim`). This is the most common case
and corresponds to the case where the target is an operator on
a n-dimensional tensor product space (here `n` would be `ndim`).
</td>
</tr><tr>
<td>
`buffer`
</td>
<td>
A buffer to store partial results in.  If not specified or None,
a new buffer is used.
</td>
</tr><tr>
<td>
`out`
</td>
<td>
The buffer to store the results in. If not specified or None, a new
buffer is used. Must have the same shape as target.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The result the conjugation.
</td>
</tr>

</table>

