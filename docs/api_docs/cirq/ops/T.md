<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.T" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.T

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



A gate that rotates around the Z axis of the Bloch sphere.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.T`, `cirq.ops.common_gates.T`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.T(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The unitary matrix of ``ZPowGate(exponent=t)`` is:

    [[1, 0],
     [0, g]]

where:

    g = exp(i·π·t).

Note in particular that this gate has a global phase factor of
e^{i·π·t/2} vs the traditionally defined rotation matrices
about the Pauli Z axis. See <a href="../../cirq/ops/rz.md"><code>cirq.rz</code></a> for rotations without the global
phase. The global phase factor can be adjusted by using the `global_shift`
parameter when initializing.

<a href="../../cirq/ops/Z.md"><code>cirq.Z</code></a>, the Pauli Z gate, is an instance of this gate at exponent=1.