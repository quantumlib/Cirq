<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.qft" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.qft

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/fourier_transform.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The quantum Fourier transform.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ops.fourier_transform.qft`, `cirq.qft`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.qft(
    *qubits,
    without_reverse: bool = False,
    inverse: bool = False
) -> "cirq.Operation"
</code></pre>



<!-- Placeholder for "Used in" -->

Transforms a qubit register from the computational basis to the frequency
basis.

The inverse quantum Fourier transform is `cirq.qft(*qubits)**-1` or
equivalently `cirq.inverse(cirq.qft(*qubits))`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The qubits to apply the qft to.
</td>
</tr><tr>
<td>
`without_reverse`
</td>
<td>
When set, swap gates at the end of the qft are omitted.
This reverses the qubit order relative to the standard qft effect,
but makes the gate cheaper to apply.
</td>
</tr><tr>
<td>
`inverse`
</td>
<td>
If set, the inverse qft is performed instead of the qft.
Equivalent to calling <a href="../../cirq/protocols/inverse.md"><code>cirq.inverse</code></a> on the result, or raising it
to the -1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a> applying the qft to the given qubits.
</td>
</tr>

</table>

