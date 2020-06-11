<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sqrt_iswap.swap_to_sqrt_iswap" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sqrt_iswap.swap_to_sqrt_iswap

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sqrt_iswap.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Implement the evolution of the hopping term using two sqrt_iswap gates

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.convert_to_sqrt_iswap.swap_to_sqrt_iswap(
    a, b, turns
)
</code></pre>



<!-- Placeholder for "Used in" -->
 and single-qubit operations. Output unitary:
[[1, 0,        0,     0],
 [0, g·c,    -i·g·s,  0],
 [0, -i·g·s,  g·c,    0],
 [0,   0,      0,     1]]
 where c = cos(theta) and s = sin(theta).
    Args:
        a: the first qubit
        b: the second qubit
        theta: The rotational angle that specifies the gate, where
        c = cos(π·t/2), s = sin(π·t/2), g = exp(i·π·t/2).