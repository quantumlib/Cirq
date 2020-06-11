<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates.decompose_phased_iswap_into_syc_precomputed" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sycamore_gates.decompose_phased_iswap_into_syc_precomputed

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decompose PhasedISwap into sycamore gates using precomputed coefficients.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sycamore_gates.decompose_phased_iswap_into_syc_precomputed(
    theta: float,
    a: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    b: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>
) -> <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This should only be called if the Gate has a phase_exponent of .25. If the
gate has an exponent of 1, decompose_phased_iswap_into_syc should be used
instead. Converting PhasedISwap gates to Sycamore is not supported if
neither of these constraints are satsified.

This synthesize a PhasedISwap in terms of four sycamore gates.  This
compilation converts the gate into a circuit involving two CZ gates, which
themselves are each represented as two Sycamore gates and single-qubit
rotations

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`theta`
</td>
<td>
rotation parameter
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

