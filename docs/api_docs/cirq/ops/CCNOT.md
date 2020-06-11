<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.CCNOT" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.CCNOT

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



A Toffoli (doubly-controlled-NOT) that can be raised to a power.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CCNOT`, `cirq.CCX`, `cirq.TOFFOLI`, `cirq.ops.CCX`, `cirq.ops.TOFFOLI`, `cirq.ops.three_qubit_gates.CCNOT`, `cirq.ops.three_qubit_gates.CCX`, `cirq.ops.three_qubit_gates.TOFFOLI`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.CCNOT(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The matrix of `CCX**t` is an 8x8 identity except the bottom right 2x2 area
is the matrix of `X**t`.