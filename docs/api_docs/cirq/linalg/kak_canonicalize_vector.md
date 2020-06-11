<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.kak_canonicalize_vector" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.kak_canonicalize_vector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Canonicalizes an XX/YY/ZZ interaction by swap/negate/shift-ing axes.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.kak_canonicalize_vector`, `cirq.linalg.decompositions.kak_canonicalize_vector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.kak_canonicalize_vector(
    x: float,
    y: float,
    z: float,
    atol: float = 1e-09
) -> <a href="../../cirq/linalg/KakDecomposition.md"><code>cirq.linalg.KakDecomposition</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
The strength of the XX interaction.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
The strength of the YY interaction.
</td>
</tr><tr>
<td>
`z`
</td>
<td>
The strength of the ZZ interaction.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
How close x2 must be to π/4 to guarantee z2 >= 0
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The canonicalized decomposition, with vector coefficients (x2, y2, z2)
</td>
</tr>
<tr>
<td>
`satisfying`
</td>
<td>
0 ≤ abs(z2) ≤ y2 ≤ x2 ≤ π/4
if x2 = π/4, z2 >= 0

Guarantees that the implied output matrix:

g · (a1 ⊗ a0) · exp(i·(x2·XX + y2·YY + z2·ZZ)) · (b1 ⊗ b0)

is approximately equal to the implied input matrix:

exp(i·(x·XX + y·YY + z·ZZ))
</td>
</tr>
</table>

