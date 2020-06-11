<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates.decompose_phased_iswap_into_syc" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sycamore_gates.decompose_phased_iswap_into_syc

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decompose PhasedISwap with an exponent of 1.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sycamore_gates.decompose_phased_iswap_into_syc(
    phase_exponent: float,
    a: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    b: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>
) -> <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This should only be called if the Gate has an exponent of 1 - otherwise,
decompose_phased_iswap_into_syc_precomputed should be used instead. The
advantage of using this function is that the resulting circuit will be
smaller.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`phase_exponent`
</td>
<td>
The exponent on the Z gates.
</td>
</tr><tr>
<td>
`a`
</td>
<td>
First qubit id to operate on
</td>
</tr><tr>
<td>
`b`
</td>
<td>
Second qubit id to operate on
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a Cirq program implementing the Phased ISWAP gate
</td>
</tr>

</table>

