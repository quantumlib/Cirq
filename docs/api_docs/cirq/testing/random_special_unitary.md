<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.random_special_unitary" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.random_special_unitary

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/lin_alg_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a random special unitary distributed with Haar measure.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.lin_alg_utils.random_special_unitary`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.random_special_unitary(
    dim: int,
    *,
    random_state: Optional[np.random.RandomState] = None
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dim`
</td>
<td>
The width and height of the matrix.
</td>
</tr><tr>
<td>
`random_state`
</td>
<td>
A seed (int) or `np.random.RandomState` class to use when
generating random values. If not set, defaults to using the module
methods in `np.random`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The sampled special unitary.
</td>
</tr>

</table>

