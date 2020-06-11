<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sqrt_iswap.fsim_gate" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sqrt_iswap.fsim_gate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sqrt_iswap.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



FSimGate has a default decomposition in cirq to XXPowGate and YYPowGate,

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sqrt_iswap.fsim_gate(
    a, b, theta, phi
)
</code></pre>



<!-- Placeholder for "Used in" -->
which is an awkward decomposition for this gate set.
Decompose into ISWAP and CZ instead.