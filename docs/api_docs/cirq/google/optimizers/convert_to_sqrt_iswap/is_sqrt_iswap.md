<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sqrt_iswap.is_sqrt_iswap" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sqrt_iswap.is_sqrt_iswap

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sqrt_iswap.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Checks if this is a ± sqrt(iSWAP) gate specified using either

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sqrt_iswap.is_sqrt_iswap(
    gate: Optional['cirq.Gate']
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->
ISwapPowGate or with the equivalent FSimGate.