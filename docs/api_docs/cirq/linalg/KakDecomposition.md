<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.KakDecomposition" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
</div>

# cirq.linalg.KakDecomposition

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A convenient description of an arbitrary two-qubit operation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.KakDecomposition`, `cirq.linalg.decompositions.KakDecomposition`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.KakDecomposition(
    *,
    global_phase: complex = complex(1),
    single_qubit_operations_before: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    interaction_coefficients: Tuple[float, float, float] = None,
    single_qubit_operations_after: Optional[Tuple[np.ndarray, np.ndarray]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Any two qubit operation U can be decomposed into the form

    U = g · (a1 ⊗ a0) · exp(i·(x·XX + y·YY + z·ZZ)) · (b1 ⊗ b0)

This class stores g, (b0, b1), (x, y, z), and (a0, a1).

#### References:

'An Introduction to Cartan's KAK Decomposition for QC Programmers'
https://arxiv.org/abs/quant-ph/0507171


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`global_phase`
</td>
<td>
g from the above equation.
</td>
</tr><tr>
<td>
`single_qubit_operations_before`
</td>
<td>
b0, b1 from the above equation.
</td>
</tr><tr>
<td>
`interaction_coefficients`
</td>
<td>
x, y, z from the above equation.
</td>
</tr><tr>
<td>
`single_qubit_operations_after`
</td>
<td>
a0, a1 from the above equation.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`global_phase`
</td>
<td>
g from the above equation.
</td>
</tr><tr>
<td>
`single_qubit_operations_before`
</td>
<td>
b0, b1 from the above equation.
</td>
</tr><tr>
<td>
`interaction_coefficients`
</td>
<td>
x, y, z from the above equation.
</td>
</tr><tr>
<td>
`single_qubit_operations_after`
</td>
<td>
a0, a1 from the above equation.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>






