<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.match_global_phase" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.match_global_phase

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/transformations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Phases the given matrices so that they agree on the phase of one entry.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.transformations.match_global_phase`, `cirq.match_global_phase`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.match_global_phase(
    a: np.ndarray,
    b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->

To maximize precision, the position with the largest entry from one of the
matrices is used when attempting to compute the phase difference between
the two matrices.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`a`
</td>
<td>
A numpy array.
</td>
</tr><tr>
<td>
`b`
</td>
<td>
Another numpy array.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
</td>
</tr>

</table>

