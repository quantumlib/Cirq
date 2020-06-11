<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.BaseDensePauliString" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__abs__"/>
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__pos__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="controlled"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="eye"/>
<meta itemprop="property" content="frozen"/>
<meta itemprop="property" content="mutable_copy"/>
<meta itemprop="property" content="num_qubits"/>
<meta itemprop="property" content="on"/>
<meta itemprop="property" content="one_hot"/>
<meta itemprop="property" content="sparse"/>
<meta itemprop="property" content="tensor_product"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="with_probability"/>
<meta itemprop="property" content="wrap_in_linear_combination"/>
<meta itemprop="property" content="I_VAL"/>
<meta itemprop="property" content="X_VAL"/>
<meta itemprop="property" content="Y_VAL"/>
<meta itemprop="property" content="Z_VAL"/>
</div>

# cirq.ops.BaseDensePauliString

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Parent class for `DensePauliString` and `MutableDensePauliString`.

Inherits From: [`Gate`](../../cirq/ops/Gate.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.BaseDensePauliString`, `cirq.ops.dense_pauli_string.BaseDensePauliString`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.BaseDensePauliString(
    pauli_mask: Union[Iterable['cirq.PAULI_GATE_LIKE'], np.ndarray],
    *,
    coefficient: Union[sympy.Basic, int, float, complex] = 1
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`pauli_mask`
</td>
<td>
A specification of the Pauli gates to use. This argument
can be a string like "IXYYZ", or a numeric list like
[0, 1, 3, 2] with I=0, X=1, Y=2, Z=3=X|Y.

The internal representation is a 1-dimensional uint8 numpy array
containing numeric values. If such a numpy array is given, and
the pauli string is mutable, the argument will be used directly
instead of being copied.
</td>
</tr><tr>
<td>
`coefficient`
</td>
<td>
A complex number. Usually +1, -1, 1j, or -1j but other
values are supported.
</td>
</tr>
</table>



## Methods

<h3 id="controlled"><code>controlled</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>controlled(
    num_controls: int = None,
    control_values: Optional[Sequence[Union[int, Collection[int]]]] = None,
    control_qid_shape: Optional[Tuple[int, ...]] = None
) -> "Gate"
</code></pre>

Returns a controlled version of this gate. If no arguments are
specified, defaults to a single qubit control.

 num_controls: Total number of control qubits.
 control_values: For which control qubit values to apply the sub
     gate.  A sequence of length `num_controls` where each
     entry is an integer (or set of integers) corresponding to the
     qubit value (or set of possible values) where that control is
     enabled.  When all controls are enabled, the sub gate is
     applied.  If unspecified, control values default to 1.
 control_qid_shape: The qid shape of the controls.  A tuple of the
     expected dimension of each control qid.  Defaults to
     `(2,) * num_controls`.  Specify this argument when using qudits.

<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>copy(
    coefficient: Optional[complex] = None,
    pauli_mask: Union[None, str, Iterable[int], np.ndarray] = None
) -> "BaseDensePauliString"
</code></pre>

Returns a copy with possibly modified contents.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`coefficient`
</td>
<td>
The new coefficient value. If not specified, defaults
to the current `coefficient` value.
</td>
</tr><tr>
<td>
`pauli_mask`
</td>
<td>
The new `pauli_mask` value. If not specified, defaults
to the current pauli mask value.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A copied instance.
</td>
</tr>

</table>



<h3 id="eye"><code>eye</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>eye(
    length: int
) -> <a href="../../cirq/ops/dense_pauli_string/TCls.md"><code>cirq.ops.dense_pauli_string.TCls</code></a>
</code></pre>

Creates a dense pauli string containing only identity gates.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`length`
</td>
<td>
The length of the dense pauli string.
</td>
</tr>
</table>



<h3 id="frozen"><code>frozen</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>frozen() -> "DensePauliString"
</code></pre>

A <a href="../../cirq/ops/DensePauliString.md"><code>cirq.DensePauliString</code></a> with the same contents.


<h3 id="mutable_copy"><code>mutable_copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mutable_copy() -> "MutableDensePauliString"
</code></pre>

A <a href="../../cirq/ops/MutableDensePauliString.md"><code>cirq.MutableDensePauliString</code></a> with the same contents.


<h3 id="num_qubits"><code>num_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_qubits() -> int
</code></pre>

The number of qubits this gate acts on.


<h3 id="on"><code>on</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on(
    *qubits
) -> "cirq.PauliString"
</code></pre>

Returns an application of this gate to the given qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*qubits`
</td>
<td>
The collection of qubits to potentially apply the gate to.
</td>
</tr>
</table>



<h3 id="one_hot"><code>one_hot</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>one_hot(
    *,
    index: int,
    length: int,
    pauli: "cirq.PAULI_GATE_LIKE"
) -> <a href="../../cirq/ops/dense_pauli_string/TCls.md"><code>cirq.ops.dense_pauli_string.TCls</code></a>
</code></pre>

Creates a dense pauli string with only one non-identity Pauli.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`index`
</td>
<td>
The index of the Pauli that is not an identity.
</td>
</tr><tr>
<td>
`pauli`
</td>
<td>
The pauli gate to put at the hot index. Can be set to either
a string ('X', 'Y', 'Z', 'I'), a cirq gate (<a href="../../cirq/ops/X.md"><code>cirq.X</code></a>,
<a href="../../cirq/ops/Y.md"><code>cirq.Y</code></a>, <a href="../../cirq/ops/Z.md"><code>cirq.Z</code></a>, or <a href="../../cirq/ops/I.md"><code>cirq.I</code></a>), or an integer (0=I, 1=X, 2=Y,
3=Z).
</td>
</tr>
</table>



<h3 id="sparse"><code>sparse</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sparse(
    qubits: Optional[Sequence['cirq.Qid']] = None
) -> "cirq.PauliString"
</code></pre>

A <a href="../../cirq/ops/PauliString.md"><code>cirq.PauliString</code></a> version of this dense pauli string.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The qubits to apply the Paulis to. Defaults to
`cirq.LineQubit.range(len(self))`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../cirq/ops/PauliString.md"><code>cirq.PauliString</code></a> with the non-identity operations from
this dense pauli string applied to appropriate qubits.
</td>
</tr>

</table>



<h3 id="tensor_product"><code>tensor_product</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tensor_product(
    other: "BaseDensePauliString"
) -> "DensePauliString"
</code></pre>

Concatenates dense pauli strings and multiplies their coefficients.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
The dense pauli string to place after the end of this one.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dense pauli string with the concatenation of the paulis from the
two input pauli strings, and the product of their coefficients.
</td>
</tr>

</table>



<h3 id="validate_args"><code>validate_args</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_args(
    qubits: Sequence['cirq.Qid']
) -> None
</code></pre>

Checks if this gate can be applied to the given qubits.

By default checks that:
- inputs are of type `Qid`
- len(qubits) == num_qubits()
- qubit_i.dimension == qid_shape[i] for all qubits

Child classes can override.  The child implementation should call
`super().validate_args(qubits)` then do custom checks.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The sequence of qubits to potentially apply the gate to.
</td>
</tr>
</table>



#### Throws:


* <b>`ValueError`</b>: The gate can't be applied to the qubits.


<h3 id="with_probability"><code>with_probability</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_probability(
    probability: "cirq.TParamVal"
) -> "cirq.Gate"
</code></pre>




<h3 id="wrap_in_linear_combination"><code>wrap_in_linear_combination</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>wrap_in_linear_combination(
    coefficient: Union[complex, float, int] = 1
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__abs__"><code>__abs__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__abs__()
</code></pre>




<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other: Union['Gate', 'cirq.LinearCombinationOfGates']
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Call self as a function.


<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    item
)
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other
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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__()
</code></pre>




<h3 id="__pos__"><code>__pos__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pos__()
</code></pre>




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    power
)
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    other
)
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other: Union['Gate', 'cirq.LinearCombinationOfGates']
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__truediv__"><code>__truediv__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/dense_pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    other
)
</code></pre>






## Class Variables

* `I_VAL = 0` <a id="I_VAL"></a>
* `X_VAL = 1` <a id="X_VAL"></a>
* `Y_VAL = 2` <a id="Y_VAL"></a>
* `Z_VAL = 3` <a id="Z_VAL"></a>
