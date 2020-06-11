<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates.find_local_equivalents" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sycamore_gates.find_local_equivalents

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Given two unitaries with the same interaction coefficients but different

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sycamore_gates.find_local_equivalents(
    unitary1: np.ndarray,
    unitary2: np.ndarray
)
</code></pre>



<!-- Placeholder for "Used in" -->
local unitary rotations determine the local unitaries that turns
one type of gate into another.

1) perform the kak decomp on each unitary and confirm interaction terms
   are equivalent
2) identify the elements of SU(2) to transform unitary2 into unitary1

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`unitary1`
</td>
<td>
unitary that we target
</td>
</tr><tr>
<td>
`unitary2`
</td>
<td>
unitary that we transform the local gates to the target
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
four 2x2 unitaries.  first two are pre-rotations last two are post
rotations.
</td>
</tr>

</table>

