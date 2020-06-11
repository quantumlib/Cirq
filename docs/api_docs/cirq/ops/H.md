<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.H" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.H

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.H`, `cirq.ops.common_gates.H`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.H(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The unitary matrix of ``HPowGate(exponent=t)`` is:

    [[g·(c-i·s/sqrt(2)), -i·g·s/sqrt(2)],
    [-i·g·s/sqrt(2)], g·(c+i·s/sqrt(2))]]

where

    c = cos(π·t/2)
    s = sin(π·t/2)
    g = exp(i·π·t/2).

Note in particular that for `t=1`, this gives the Hadamard matrix.

<a href="../../cirq/ops/H.md"><code>cirq.H</code></a>, the Hadamard gate, is an instance of this gate at `exponent=1`.