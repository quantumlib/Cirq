<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.CZ" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.CZ

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



A gate that applies a phase to the |11⟩ state of two qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CZ`, `cirq.ops.common_gates.CZ`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.CZ(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The unitary matrix of `CZPowGate(exponent=t)` is:

    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, g]]

where:

    g = exp(i·π·t).

<a href="../../cirq/ops/CZ.md"><code>cirq.CZ</code></a>, the controlled Z gate, is an instance of this gate at
`exponent=1`.