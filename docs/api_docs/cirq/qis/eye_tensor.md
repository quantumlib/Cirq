<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.eye_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.eye_tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns an identity matrix reshaped into a tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.eye_tensor`, `cirq.qis.states.eye_tensor`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.eye_tensor(
    half_shape: Tuple[int, ...],
    *,
    dtype: Type[np.number]
) -> np.array
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`half_shape`
</td>
<td>
A tuple representing the number of quantum levels of each
qubit the returned matrix applies to.  `half_shape` is (2, 2, 2) for
a three-qubit identity operation tensor.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The numpy dtype of the new array.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The created numpy array with shape `half_shape + half_shape`.
</td>
</tr>

</table>

