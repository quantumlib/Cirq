<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.vis.heatmap.relative_luminance" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.vis.heatmap.relative_luminance

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/vis/heatmap.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the relative luminance according to W3C specification.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.vis.heatmap.relative_luminance(
    color: np.ndarray
) -> float
</code></pre>



<!-- Placeholder for "Used in" -->

Spec: https://www.w3.org/TR/WCAG21/#dfn-relative-luminance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`color`
</td>
<td>
a numpy array with the first 3 elements red, green, and blue
with values in [0, 1].
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
relative luminance of color in [0, 1].
</td>
</tr>

</table>

