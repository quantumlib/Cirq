<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates.cphase" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sycamore_gates.cphase

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implement a cphase using the Ising gate generated from 2 Sycamore gates

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sycamore_gates.cphase(
    theta: float,
    q0: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>,
    q1: <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>
) -> <a href="../../../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

A CPHASE gate has the matrix diag([1, 1, 1, exp(1j * theta)]) and can
be mapped to the Ising gate by prep and post rotations of Z-pi/4.
We drop the global phase shift of theta/4.

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


returns:
    a cirq program implementating cphase