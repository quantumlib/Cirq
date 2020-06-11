<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.givens" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.givens

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/phased_iswap_gate.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns gate with matrix exp(-i angle_rads (Y⊗X - X⊗Y) / 2).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.givens`, `cirq.ops.phased_iswap_gate.givens`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.givens(
    angle_rads: <a href="../../cirq/value/TParamVal.md"><code>cirq.value.TParamVal</code></a>
) -> <a href="../../cirq/ops/PhasedISwapPowGate.md"><code>cirq.ops.PhasedISwapPowGate</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

In numerical linear algebra Givens rotation is any linear transformation
with matrix equal to the identity except for a 2x2 orthogonal submatrix
[[cos(a), -sin(a)], [sin(a), cos(a)]] which performs a 2D rotation on a
subspace spanned by two basis vectors. In quantum computational chemistry
the term is used to refer to the two-qubit gate defined as

    givens(a) ≡ exp(-i a (Y⊗X - X⊗Y) / 2)

with the matrix

    [[1, 0, 0, 0],
     [0, c, -s, 0],
     [0, s, c, 0],
     [0, 0, 0, 1]]

where

    c = cos(a),
    s = sin(a).

The matrix is a Givens rotation in the numerical linear algebra sense
acting on the subspace spanned by the |01⟩ and |10⟩ states.

The gate is also equivalent to the ISWAP conjugated by T^-1 ⊗ T.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`angle_rads`
</td>
<td>
The rotation angle in radians.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A phased iswap gate for the given rotation.
</td>
</tr>

</table>

