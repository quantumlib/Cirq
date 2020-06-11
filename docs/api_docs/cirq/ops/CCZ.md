<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.CCZ" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.CCZ

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



A doubly-controlled-Z that can be raised to a power.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CCZ`, `cirq.ops.three_qubit_gates.CCZ`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.CCZ(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The matrix of `CCZ**t` is `diag(1, 1, 1, 1, 1, 1, 1, exp(i pi t))`.