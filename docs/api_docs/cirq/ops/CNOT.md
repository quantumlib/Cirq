<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.CNOT" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.CNOT

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



A gate that applies a controlled power of an X gate.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CNOT`, `cirq.CX`, `cirq.ops.CX`, `cirq.ops.common_gates.CNOT`, `cirq.ops.common_gates.CX`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.CNOT(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

When applying CNOT (controlled-not) to qubits, you can either use
positional arguments CNOT(q1, q2), where q2 is toggled when q1 is on,
or named arguments CNOT(control=q1, target=q2).
(Mixing the two is not permitted.)

The unitary matrix of `CXPowGate(exponent=t)` is:

    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, g·c, -i·g·s],
     [0, 0, -i·g·s, g·c]]

where:

    c = cos(π·t/2)
    s = sin(π·t/2)
    g = exp(i·π·t/2).

<a href="../../cirq/ops/CNOT.md"><code>cirq.CNOT</code></a>, the controlled NOT gate, is an instance of this gate at
`exponent=1`.