<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.SYC" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.SYC

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



The Sycamore gate is a two-qubit gate equivalent to FSimGate(π/2, π/6).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.ops.SYC`, `cirq.google.ops.sycamore_gate.SYC`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.SYC(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The unitary of this gate is

    [[1, 0, 0, 0],
     [0, 0, -1j, 0],
     [0, -1j, 0, 0],
     [0, 0, 0, exp(- 1j * π/6)]]

This gate can be performed on the Google's Sycamore chip and
is close to the gates that were used to demonstrate quantum
supremacy used in this paper:
https://www.nature.com/articles/s41586-019-1666-5