<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.study.plot_state_histogram" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.study.plot_state_histogram

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/study/visualize.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Plot the state histogram from a single result with repetitions.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.plot_state_histogram`, `cirq.study.visualize.plot_state_histogram`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.study.plot_state_histogram(
    result: <a href="../../cirq/study/TrialResult.md"><code>cirq.study.TrialResult</code></a>
) -> np.ndarray
</code></pre>



<!-- Placeholder for "Used in" -->

States is a bitstring representation of all the qubit states in a single
result.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`result`
</td>
<td>
The trial results to plot.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The histogram. A list of values plotted on the y-axis.
</td>
</tr>

</table>

