<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.qis.one_hot" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.qis.one_hot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/qis/states.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a numpy array with all 0s and a single non-zero entry(default 1).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.one_hot`, `cirq.qis.states.one_hot`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.qis.one_hot(
    *,
    index: Union[None, int, Sequence[int]] = None,
    shape: Union[int, Sequence[int]] = 1,
    value: Any = 1,
    dtype: Type[np.number]
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`index`
</td>
<td>
The index that should store the `value` argument instead of 0.
If not specified, defaults to the start of the array.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
The shape of the array.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
The hot value to place at `index` in the result.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
The dtype of the array.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The created numpy array.
</td>
</tr>

</table>

