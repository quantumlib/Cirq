<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.PauliString" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__pos__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__rpow__"/>
<meta itemprop="property" content="__rsub__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="conjugated_by"/>
<meta itemprop="property" content="controlled_by"/>
<meta itemprop="property" content="dense"/>
<meta itemprop="property" content="equal_up_to_coefficient"/>
<meta itemprop="property" content="expectation_from_density_matrix"/>
<meta itemprop="property" content="expectation_from_state_vector"/>
<meta itemprop="property" content="expectation_from_wavefunction"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="items"/>
<meta itemprop="property" content="keys"/>
<meta itemprop="property" content="map_qubits"/>
<meta itemprop="property" content="pass_operations_over"/>
<meta itemprop="property" content="to_z_basis_ops"/>
<meta itemprop="property" content="transform_qubits"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="values"/>
<meta itemprop="property" content="with_probability"/>
<meta itemprop="property" content="with_qubits"/>
<meta itemprop="property" content="with_tags"/>
<meta itemprop="property" content="zip_items"/>
<meta itemprop="property" content="zip_paulis"/>
</div>

# cirq.ops.PauliString

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An effect applied to a collection of qubits.

Inherits From: [`Operation`](../../cirq/ops/Operation.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.PauliString`, `cirq.ops.pauli_string.PauliString`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.PauliString(
    qubit_pauli_map: Optional[Dict['cirq.Qid', 'cirq.Pauli']] = None,
    *contents,
    coefficient: Union[int, float, complex] = 1
)
</code></pre>



<!-- Placeholder for "Used in" -->

The most common kind of Operation is a GateOperation, which separates its
effect into a qubit-independent Gate and the qubits it should be applied to.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`*contents`
</td>
<td>
A value or values to convert into a pauli string. This
can be a number, a pauli operation, a dictionary from qubit to
pauli/identity gates, or collections thereof. If a list of
values is given, they are each individually converted and then
multiplied from left to right in order.
</td>
</tr><tr>
<td>
`qubit_pauli_map`
</td>
<td>
Initial dictionary mapping qubits to pauli
operations. Defaults to the empty dictionary. Note that, unlike
dictionaries passed to contents, this dictionary must not
contain any identity gate values. Further note that this
argument specifies values that are logically *before* factors
specified in `contents`; `contents` are *right* multiplied onto
the values in this dictionary.
</td>
</tr><tr>
<td>
`coefficient`
</td>
<td>
Initial scalar coefficient. Defaults to 1.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`coefficient`
</td>
<td>

</td>
</tr><tr>
<td>
`gate`
</td>
<td>

</td>
</tr><tr>
<td>
`qubits`
</td>
<td>

</td>
</tr><tr>
<td>
`tags`
</td>
<td>
Returns a tuple of the operation's tags.
</td>
</tr><tr>
<td>
`untagged`
</td>
<td>
Returns the underlying operation without any tags.
</td>
</tr>
</table>



## Methods

<h3 id="conjugated_by"><code>conjugated_by</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>conjugated_by(
    clifford: "cirq.OP_TREE"
) -> "PauliString"
</code></pre>

Returns the Pauli string conjugated by a clifford operation.

The product-of-Paulis $P$ conjugated by the Clifford operation $C$ is

    $$
    C^\dagger P C
    $$

For example, conjugating a +Y operation by an S operation results in a
+X operation (as opposed to a -X operation).

In a circuit diagram where `P` is a pauli string observable immediately
after a Clifford operation `C`, the pauli string `P.conjugated_by(C)` is
the equivalent pauli string observable just before `C`.

    --------------------------C---P---

    = ---C---P------------------------

    = ---C---P---------C^-1---C-------

    = ---C---P---C^-1---------C-------

    = --(C^-1 · P · C)--------C-------

    = ---P.conjugated_by(C)---C-------

Analogously, a Pauli product P can be moved from before a Clifford C in
a circuit diagram to after the Clifford C by conjugating P by the
inverse of C:

    ---P---C---------------------------

    = -----C---P.conjugated_by(C^-1)---

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`clifford`
</td>
<td>
The Clifford operation to conjugate by. This can be an
individual operation, or a tree of operations.

Note that the composite Clifford operation defined by a sequence
of operations is equivalent to a circuit containing those
operations in the given order. Somewhat counter-intuitively,
this means that the operations in the sequence are conjugated
onto the Pauli string in reverse order. For example,
`P.conjugated_by([C1, C2])` is equivalent to
`P.conjugated_by(C2).conjugated_by(C1)`.
</td>
</tr>
</table>



#### Examples:

>>> a, b = cirq.LineQubit.range(2)
>>> print(cirq.X(a).conjugated_by(cirq.CZ(a, b)))
X(0)*Z(1)
>>> print(cirq.X(a).conjugated_by(cirq.S(a)))
-Y(0)
>>> print(cirq.X(a).conjugated_by([cirq.H(a), cirq.CNOT(a, b)]))
Z(0)*X(1)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The Pauli string conjugated by the given Clifford operation.
</td>
</tr>

</table>



<h3 id="controlled_by"><code>controlled_by</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>controlled_by(
    control_values: Optional[Sequence[Union[int, Collection[int]]]] = None,
    *control_qubits
) -> "cirq.Operation"
</code></pre>

Returns a controlled version of this operation. If no control_qubits
   are specified, returns self.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`control_qubits`
</td>
<td>
Qubits to control the operation by. Required.
</td>
</tr><tr>
<td>
`control_values`
</td>
<td>
For which control qubit values to apply the
operation.  A sequence of the same length as `control_qubits`
where each entry is an integer (or set of integers)
corresponding to the qubit value (or set of possible values)
where that control is enabled.  When all controls are enabled,
the operation is applied.  If unspecified, control values
default to 1.
</td>
</tr>
</table>



<h3 id="dense"><code>dense</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dense(
    qubits: Sequence['cirq.Qid']
) -> "cirq.DensePauliString"
</code></pre>

Returns a <a href="../../cirq/ops/DensePauliString.md"><code>cirq.DensePauliString</code></a> version of this Pauli string.

This method satisfies the invariant `P.dense(qubits).on(*qubits) == P`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The implicit sequence of qubits used by the dense pauli
string. Specifically, if the returned dense Pauli string was
applied to these qubits (via its `on` method) then the result
would be a Pauli string equivalent to the receiving Pauli
string.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../cirq/ops/DensePauliString.md"><code>cirq.DensePauliString</code></a> instance `D` such that `D.on(*qubits)`
equals the receiving <a href="../../cirq/ops/PauliString.md"><code>cirq.PauliString</code></a> instance `P`.
</td>
</tr>

</table>



<h3 id="equal_up_to_coefficient"><code>equal_up_to_coefficient</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>equal_up_to_coefficient(
    other: "PauliString"
) -> bool
</code></pre>




<h3 id="expectation_from_density_matrix"><code>expectation_from_density_matrix</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expectation_from_density_matrix(
    state: np.ndarray,
    qubit_map: Mapping[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, int],
    *,
    atol: float = 1e-07,
    check_preconditions: bool = True
) -> float
</code></pre>

Evaluate the expectation of this PauliString given a density matrix.

Compute the expectation value of this PauliString with respect to an
array representing a density matrix. By convention expectation values
are defined for Hermitian operators, and so this method will fail if
this PauliString is non-Hermitian.

`state` must be an array representation of a density matrix and have
shape `(2 ** n, 2 ** n)` or `(2, 2, ..., 2)` (2*n entries), where
`state` is expressed over n qubits.

`qubit_map` must assign an integer index to each qubit in this
PauliString that determines which bit position of a computational basis
state that qubit corresponds to. For example if `state` represents
$|0\rangle |+\rangle$ and `q0, q1 = cirq.LineQubit.range(2)` then:

    cirq.X(q0).expectation(state, qubit_map={q0: 0, q1: 1}) = 0
    cirq.X(q0).expectation(state, qubit_map={q0: 1, q1: 0}) = 1

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
A map from all qubits used in this PauliString to the
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
NotImplementedError if this PauliString is non-Hermitian.
</td>
</tr>

</table>



<h3 id="expectation_from_state_vector"><code>expectation_from_state_vector</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>expectation_from_state_vector(
    state_vector: np.ndarray,
    qubit_map: Mapping[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, int],
    *,
    atol: float = 1e-07,
    check_preconditions: bool = True
) -> float
</code></pre>

Evaluate the expectation of this PauliString given a state vector.

Compute the expectation value of this PauliString with respect to a
state vector. By convention expectation values are defined for Hermitian
operators, and so this method will fail if this PauliString is
non-Hermitian.

`state` must be an array representation of a state vector and have
shape `(2 ** n, )` or `(2, 2, ..., 2)` (n entries) where `state` is
expressed over n qubits.

`qubit_map` must assign an integer index to each qubit in this
PauliString that determines which bit position of a computational basis
state that qubit corresponds to. For example if `state` represents
$|0\rangle |+\rangle$ and `q0, q1 = cirq.LineQubit.range(2)` then:

    cirq.X(q0).expectation(state, qubit_map={q0: 0, q1: 1}) = 0
    cirq.X(q0).expectation(state, qubit_map={q0: 1, q1: 0}) = 1

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`state_vector`
</td>
<td>
An array representing a valid state vector.
</td>
</tr><tr>
<td>
`qubit_map`
</td>
<td>
A map from all qubits used in this PauliString to the
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



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
NotImplementedError if this PauliString is non-Hermitian.
</td>
</tr>

</table>



<h3 id="expectation_from_wavefunction"><code>expectation_from_wavefunction</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

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

Use expectation_from_state_vector instead

<h3 id="get"><code>get</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    key: "cirq.Qid",
    default=None
)
</code></pre>




<h3 id="items"><code>items</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>items() -> ItemsView
</code></pre>




<h3 id="keys"><code>keys</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>keys() -> KeysView[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]
</code></pre>




<h3 id="map_qubits"><code>map_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map_qubits(
    qubit_map: Dict[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, <a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]
) -> "PauliString"
</code></pre>




<h3 id="pass_operations_over"><code>pass_operations_over</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pass_operations_over(
    ops: Iterable['cirq.Operation'],
    after_to_before: bool = False
) -> "PauliString"
</code></pre>

Determines how the Pauli string changes when conjugated by Cliffords.

The output and input pauli strings are related by a circuit equivalence.
In particular, this circuit:

    ───ops───INPUT_PAULI_STRING───

will be equivalent to this circuit:

    ───OUTPUT_PAULI_STRING───ops───

up to global phase (assuming `after_to_before` is not set).

If ops together have matrix C, the Pauli string has matrix P, and the
output Pauli string has matrix P', then P' == C^-1 P C up to
global phase.

Setting `after_to_before` inverts the relationship, so that the output
is the input and the input is the output. Equivalently, it inverts C.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`ops`
</td>
<td>
The operations to move over the string.
</td>
</tr><tr>
<td>
`after_to_before`
</td>
<td>
Determines whether the operations start after the
pauli string, instead of before (and so are moving in the
opposite direction).
</td>
</tr>
</table>



<h3 id="to_z_basis_ops"><code>to_z_basis_ops</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_z_basis_ops() -> Iterator[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]
</code></pre>

Returns operations to convert the qubits to the computational basis.
        

<h3 id="transform_qubits"><code>transform_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>transform_qubits(
    func: Callable[['cirq.Qid'], 'cirq.Qid']
) -> "Operation"
</code></pre>

Returns the same operation, but with different qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`func`
</td>
<td>
The function to use to turn each current qubit into a desired
new qubit.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The receiving operation but with qubits transformed by the given
function.
</td>
</tr>

</table>



<h3 id="validate_args"><code>validate_args</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_args(
    qubits: Sequence['cirq.Qid']
)
</code></pre>

Raises an exception if the `qubits` don't match this operation's qid
shape.

Call this method from a subclass's `with_qubits` method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`qubits`
</td>
<td>
The new qids for the operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
The operation had qids that don't match it's qid shape.
</td>
</tr>
</table>



<h3 id="values"><code>values</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>values() -> ValuesView[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>]
</code></pre>




<h3 id="with_probability"><code>with_probability</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_probability(
    probability: "cirq.TParamVal"
) -> "cirq.Operation"
</code></pre>




<h3 id="with_qubits"><code>with_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_qubits(
    *new_qubits
) -> "PauliString"
</code></pre>

Returns the same operation, but applied to different qubits.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`new_qubits`
</td>
<td>
The new qubits to apply the operation to. The order must
exactly match the order of qubits returned from the operation's
`qubits` property.
</td>
</tr>
</table>



<h3 id="with_tags"><code>with_tags</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_tags(
    *new_tags
) -> "cirq.TaggedOperation"
</code></pre>

Creates a new TaggedOperation, with this op and the specified tags.

This method can be used to attach meta-data to specific operations
without affecting their functionality.  The intended usage is to
attach classes intended for this purpose or strings to mark operations
for specific usage that will be recognized by consumers.  Specific
examples include ignoring this operation in optimization passes,
hardware-specific functionality, or circuit diagram customizability.

Tags can be a list of any type of object that is useful to identify
this operation as long as the type is hashable.  If you wish the
resulting operation to be eventually serialized into JSON, you should
also restrict the operation to be JSON serializable.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`new_tags`
</td>
<td>
The tags to wrap this operation in.
</td>
</tr>
</table>



<h3 id="zip_items"><code>zip_items</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>zip_items(
    other: "PauliString"
) -> Iterator[Tuple[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>, Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, <a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>]]]
</code></pre>




<h3 id="zip_paulis"><code>zip_paulis</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>zip_paulis(
    other: "PauliString"
) -> Iterator[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, <a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>]]
</code></pre>




<h3 id="__add__"><code>__add__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    other
)
</code></pre>




<h3 id="__bool__"><code>__bool__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>




<h3 id="__contains__"><code>__contains__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    key: "cirq.Qid"
) -> bool
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    key: "cirq.Qid"
) -> <a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>
</code></pre>




<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__() -> Iterator[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__() -> int
</code></pre>




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other
) -> "PauliString"
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__neg__"><code>__neg__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__() -> "PauliString"
</code></pre>




<h3 id="__pos__"><code>__pos__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pos__() -> "PauliString"
</code></pre>




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    power
)
</code></pre>




<h3 id="__radd__"><code>__radd__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    other
)
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    other
) -> "PauliString"
</code></pre>




<h3 id="__rpow__"><code>__rpow__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rpow__(
    base
)
</code></pre>




<h3 id="__rsub__"><code>__rsub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rsub__(
    other
)
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other
)
</code></pre>




<h3 id="__truediv__"><code>__truediv__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/pauli_string.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    other
)
</code></pre>






