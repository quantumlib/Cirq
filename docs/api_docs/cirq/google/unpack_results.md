<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.unpack_results" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.unpack_results

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v1/programs.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Unpack data from a bitstring into individual measurement results.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v1.programs.unpack_results`, `cirq.google.api.v1.unpack_results`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.unpack_results(
    data: bytes,
    repetitions: int,
    key_sizes: Sequence[Tuple[str, int]]
) -> Dict[str, np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
Packed measurement results, in the form <rep0><rep1>...
where each repetition is <key0_0>..<key0_{size0-1}><key1_0>...
with bits packed in little-endian order in each byte.
</td>
</tr><tr>
<td>
`repetitions`
</td>
<td>
number of repetitions.
</td>
</tr><tr>
<td>
`key_sizes`
</td>
<td>
Keys and sizes of the measurements in the data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Dict mapping measurement key to a 2D array of boolean results. Each
array has shape (repetitions, size) with size for that measurement.
</td>
</tr>

</table>

