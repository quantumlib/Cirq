<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.XX" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.XX

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



The X-parity gate, possibly raised to a power.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.XX`, `cirq.ops.parity_gates.XX`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.XX(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

At exponent=1, this gate implements the following unitary:

    X⊗X = [0 0 0 1]
          [0 0 1 0]
          [0 1 0 0]
          [1 0 0 0]

See also: `cirq.MSGate` (the Mølmer–Sørensen gate), which is implemented via
    this class.