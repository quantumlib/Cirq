<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.partial_trace" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.partial_trace

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/transformations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Takes the partial trace of a given tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.transformations.partial_trace`, `cirq.partial_trace`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.partial_trace(
    tensor: np.ndarray,
    keep_indices: List[int]
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

The input tensor must have shape `(d_0, ..., d_{k-1}, d_0, ..., d_{k-1})`.
The trace is done over all indices that are not in keep_indices. The
resulting tensor has shape `(d_{i_0}, ..., d_{i_r}, d_{i_0}, ..., d_{i_r})`
where `i_j` is the `j`th element of `keep_indices`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
The tensor to sum over. This tensor must have a shape
`(d_0, ..., d_{k-1}, d_0, ..., d_{k-1})`.
</td>
</tr><tr>
<td>
`keep_indices`
</td>
<td>
Which indices to not sum over. These are only the indices
of the first half of the tensors indices (i.e. all elements must
be between `0` and `tensor.ndims / 2 - 1` inclusive).
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
if the tensor is not of the correct shape or the indices
are not from the first half of valid indices for the tensor.
</td>
</tr>
</table>

