<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.SWAP" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.SWAP

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



The SWAP gate, possibly raised to a power. Exchanges qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SWAP`, `cirq.ops.common_gates.SWAP`, `cirq.ops.swap_gates.SWAP`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.SWAP(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

SwapPowGate()**t = SwapPowGate(exponent=t) and acts on two qubits in the
computational basis as the matrix:

    [[1, 0, 0, 0],
     [0, g·c, -i·g·s, 0],
     [0, -i·g·s, g·c, 0],
     [0, 0, 0, 1]]

where:

    c = cos(π·t/2)
    s = sin(π·t/2)
    g = exp(i·π·t/2).

<a href="../../cirq/ops/SWAP.md"><code>cirq.SWAP</code></a>, the swap gate, is an instance of this gate at exponent=1.