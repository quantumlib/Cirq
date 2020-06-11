<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sqrt_iswap.iswap_to_sqrt_iswap" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.convert_to_sqrt_iswap.iswap_to_sqrt_iswap

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
<code>cirq.google.optimizers.convert_to_sqrt_iswap.iswap_to_sqrt_iswap(
    a, b, turns
)
</code></pre>



<!-- Placeholder for "Used in" -->
 and single-qubit operations. Output unitary:
[1   0   0   0],
[0   c  is   0],
[0  is   c   0],
[0   0   0   1],
where c = cos(t * np.pi / 2) and s = sin(t * np.pi / 2).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`a`
</td>
<td>
the first qubit
</td>
</tr><tr>
<td>
`b`
</td>
<td>
the second qubit
</td>
</tr><tr>
<td>
`t`
</td>
<td>
Exponent that specifies the evolution time in number of rotations.
</td>
</tr>
</table>

