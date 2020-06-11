<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates.swap_rzz" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sycamore_gates.swap_rzz

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An implementation of SWAP * EXP(1j theta ZZ) using three sycamore gates.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sycamore_gates.swap_rzz(
    theta: float,
    q0: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    q1: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>
) -> <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

This builds off of the zztheta method.  A sycamore gate following the
zz-gate is a SWAP EXP(1j (THETA - pi/24) ZZ).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`theta`
</td>
<td>
exp(1j * theta )
</td>
</tr><tr>
<td>
`q0`
</td>
<td>
First qubit id to operate on
</td>
</tr><tr>
<td>
`q1`
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
The circuit that implements ZZ followed by a swap
</td>
</tr>

</table>

