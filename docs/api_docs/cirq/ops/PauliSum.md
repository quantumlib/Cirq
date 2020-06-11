<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.PauliSum" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__rsub__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="expectation_from_density_matrix"/>
<meta itemprop="property" content="expectation_from_state_vector"/>
<meta itemprop="property" content="expectation_from_wavefunction"/>
<meta itemprop="property" content="from_pauli_strings"/>
<meta itemprop="property" content="wrap"/>
</div>

# cirq.ops.PauliSum

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Represents operator defined by linear combination of PauliStrings.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.PauliSum`, `cirq.ops.linear_combinations.PauliSum`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.PauliSum(
    linear_dict: Optional[value.LinearDict[UnitPauliStringT]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Since PauliStrings store their own coefficients, this class
does not implement the LinearDict interface. Instead, you can
add and subtract terms and then iterate over the resulting
(simplified) expression.

Under the hood, this class is backed by a LinearDict with coefficient-less
PauliStrings as keys. PauliStrings are reconstructed on-the-fly during
iteration.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`qubits`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy() -> "PauliSum"
</code></pre>




<h3 id="expectation_from_density_matrix"><code>expectation_from_density_matrix</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expectation_from_density_matrix(
    state: np.ndarray,
    qubit_map: Mapping[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, int],
    *,
    atol: float = 1e-07,
    check_preconditions: bool = True
) -> float
</code></pre>

Evaluate the expectation of this PauliSum given a density matrix.

See <a href="../../cirq/ops/PauliString.md#expectation_from_density_matrix"><code>PauliString.expectation_from_density_matrix</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`state`
</td>
<td>
An array representing a valid  density matrix.
</td>
</tr><tr>
<td>
`qubit_map`
</td>
<td>
A map from all qubits used in this PauliSum to the
indices of the qubits that `state` is defined over.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute numerical tolerance.
</td>
</tr><tr>
<td>
`check_preconditions`
</td>
<td>
Whether to check that `state` represents a
valid density matrix.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The expectation value of the input state.
</td>
</tr>

</table>



<h3 id="expectation_from_state_vector"><code>expectation_from_state_vector</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expectation_from_state_vector(
    state_vector: np.ndarray,
    qubit_map: Mapping[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, int],
    *,
    atol: float = 1e-07,
    check_preconditions: bool = True
) -> float
</code></pre>

Evaluate the expectation of this PauliSum given a state vector.

See <a href="../../cirq/ops/PauliString.md#expectation_from_state_vector"><code>PauliString.expectation_from_state_vector</code></a>.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`state`
</td>
<td>
An array representing a valid state vector.
</td>
</tr><tr>
<td>
`qubit_map`
</td>
<td>
A map from all qubits used in this PauliSum to the
indices of the qubits that `state_vector` is defined over.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute numerical tolerance.
</td>
</tr><tr>
<td>
`check_preconditions`
</td>
<td>
Whether to check that `state_vector` represents
a valid state vector.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The expectation value of the input state.
</td>
</tr>

</table>



<h3 id="expectation_from_wavefunction"><code>expectation_from_wavefunction</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expectation_from_wavefunction(
    state: np.ndarray,
    qubit_map: Mapping[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, int],
    *,
    atol: float = 1e-07,
    check_preconditions: bool = True
) -> float
</code></pre>

THIS FUNCTION IS DEPRECATED.

IT WILL BE REMOVED IN `cirq v0.10.0`.

Use expectation_from_state_vector instead.

<h3 id="from_pauli_strings"><code>from_pauli_strings</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_pauli_strings(
    terms: Union[<a href="../../cirq/ops/PauliString.md"><code>cirq.ops.PauliString</code></a>, List[<a href="../../cirq/ops/PauliString.md"><code>cirq.ops.PauliString</code></a>]]
) -> "PauliSum"
</code></pre>




<h3 id="wrap"><code>wrap</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>wrap(
    val: <a href="../../cirq/ops/PauliSumLike.md"><code>cirq.ops.PauliSumLike</code></a>
) -> "PauliSum"
</code></pre>




<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>




<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__() -> bool
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other: <a href="../../cirq/ops/PauliSumLike.md"><code>cirq.ops.PauliSumLike</code></a>
)
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__neg__"><code>__neg__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__()
</code></pre>




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    exponent: int
)
</code></pre>




<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other
)
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    other: <a href="../../cirq/ops/PauliSumLike.md"><code>cirq.ops.PauliSumLike</code></a>
)
</code></pre>




<h3 id="__rsub__"><code>__rsub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rsub__(
    other
)
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other
)
</code></pre>




<h3 id="__truediv__"><code>__truediv__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/linear_combinations.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    a: <a href="../../cirq/value/Scalar.md"><code>cirq.value.Scalar</code></a>
)
</code></pre>






