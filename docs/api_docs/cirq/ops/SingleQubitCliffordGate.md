<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.SingleQubitCliffordGate" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="H"/>
<meta itemprop="property" content="I"/>
<meta itemprop="property" content="X"/>
<meta itemprop="property" content="X_nsqrt"/>
<meta itemprop="property" content="X_sqrt"/>
<meta itemprop="property" content="Y"/>
<meta itemprop="property" content="Y_nsqrt"/>
<meta itemprop="property" content="Y_sqrt"/>
<meta itemprop="property" content="Z"/>
<meta itemprop="property" content="Z_nsqrt"/>
<meta itemprop="property" content="Z_sqrt"/>
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="commutes_with_pauli"/>
<meta itemprop="property" content="commutes_with_single_qubit_gate"/>
<meta itemprop="property" content="controlled"/>
<meta itemprop="property" content="decompose_rotation"/>
<meta itemprop="property" content="equivalent_gate_before"/>
<meta itemprop="property" content="from_double_map"/>
<meta itemprop="property" content="from_pauli"/>
<meta itemprop="property" content="from_quarter_turns"/>
<meta itemprop="property" content="from_single_map"/>
<meta itemprop="property" content="from_unitary"/>
<meta itemprop="property" content="from_xz_map"/>
<meta itemprop="property" content="merged_with"/>
<meta itemprop="property" content="num_qubits"/>
<meta itemprop="property" content="on"/>
<meta itemprop="property" content="on_each"/>
<meta itemprop="property" content="transform"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="with_probability"/>
<meta itemprop="property" content="wrap_in_linear_combination"/>
</div>

# cirq.ops.SingleQubitCliffordGate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Any single qubit Clifford rotation.

Inherits From: [`SingleQubitGate`](../../cirq/ops/SingleQubitGate.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SingleQubitCliffordGate`, `cirq.ops.clifford_gate.SingleQubitCliffordGate`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.SingleQubitCliffordGate(
    *,
    _rotation_map: Dict[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, <a href="../../cirq/ops/PauliTransform.md"><code>cirq.ops.PauliTransform</code></a>],
    _inverse_map: Dict[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, <a href="../../cirq/ops/PauliTransform.md"><code>cirq.ops.PauliTransform</code></a>]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="H"><code>H</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>H(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="I"><code>I</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>I(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="X"><code>X</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>X(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="X_nsqrt"><code>X_nsqrt</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>X_nsqrt(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="X_sqrt"><code>X_sqrt</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>X_sqrt(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="Y"><code>Y</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Y(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="Y_nsqrt"><code>Y_nsqrt</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Y_nsqrt(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="Y_sqrt"><code>Y_sqrt</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Y_sqrt(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="Z"><code>Z</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Z(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="Z_nsqrt"><code>Z_nsqrt</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Z_nsqrt(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="Z_sqrt"><code>Z_sqrt</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>Z_sqrt(
    *args, **kwargs
)
</code></pre>

Any single qubit Clifford rotation.


<h3 id="commutes_with_pauli"><code>commutes_with_pauli</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>commutes_with_pauli(
    pauli: <a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>
) -> bool
</code></pre>




<h3 id="commutes_with_single_qubit_gate"><code>commutes_with_single_qubit_gate</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>commutes_with_single_qubit_gate(
    gate: "SingleQubitCliffordGate"
) -> bool
</code></pre>

Tests if the two circuits would be equivalent up to global phase:
--self--gate-- and --gate--self--

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

<h3 id="decompose_rotation"><code>decompose_rotation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>decompose_rotation() -> Sequence[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, int]]
</code></pre>

Returns ((first_rotation_axis, first_rotation_quarter_turns), ...)

This is a sequence of zero, one, or two rotations.

<h3 id="equivalent_gate_before"><code>equivalent_gate_before</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>equivalent_gate_before(
    after: "SingleQubitCliffordGate"
) -> "SingleQubitCliffordGate"
</code></pre>

Returns a SingleQubitCliffordGate such that the circuits
    --output--self-- and --self--gate--
are equivalent up to global phase.

<h3 id="from_double_map"><code>from_double_map</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_double_map(
    pauli_map_to: Optional[Dict[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]]] = None,
    *,
    x_to: Optional[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]] = None,
    y_to: Optional[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]] = None,
    z_to: Optional[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]] = None
) -> "SingleQubitCliffordGate"
</code></pre>

Returns a SingleQubitCliffordGate for the
specified transform with a 90 or 180 degree rotation.

Either pauli_map_to or two of (x_to, y_to, z_to) may be specified.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`pauli_map_to`
</td>
<td>
A dictionary with two key value pairs describing
two transforms.
</td>
</tr><tr>
<td>
`x_to`
</td>
<td>
The transform from cirq.X
</td>
</tr><tr>
<td>
`y_to`
</td>
<td>
The transform from cirq.Y
</td>
</tr><tr>
<td>
`z_to`
</td>
<td>
The transform from cirq.Z
</td>
</tr>
</table>



<h3 id="from_pauli"><code>from_pauli</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_pauli(
    pauli: <a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>,
    sqrt: bool = False
) -> "SingleQubitCliffordGate"
</code></pre>




<h3 id="from_quarter_turns"><code>from_quarter_turns</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_quarter_turns(
    pauli: <a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>,
    quarter_turns: int
) -> "SingleQubitCliffordGate"
</code></pre>




<h3 id="from_single_map"><code>from_single_map</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_single_map(
    pauli_map_to: Optional[Dict[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]]] = None,
    *,
    x_to: Optional[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]] = None,
    y_to: Optional[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]] = None,
    z_to: Optional[Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]] = None
) -> "SingleQubitCliffordGate"
</code></pre>

Returns a SingleQubitCliffordGate for the
specified transform with a 90 or 180 degree rotation.

The arguments are exclusive, only one may be specified.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`pauli_map_to`
</td>
<td>
A dictionary with a single key value pair describing
the transform.
</td>
</tr><tr>
<td>
`x_to`
</td>
<td>
The transform from cirq.X
</td>
</tr><tr>
<td>
`y_to`
</td>
<td>
The transform from cirq.Y
</td>
</tr><tr>
<td>
`z_to`
</td>
<td>
The transform from cirq.Z
</td>
</tr>
</table>



<h3 id="from_unitary"><code>from_unitary</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_unitary(
    u: np.ndarray
) -> Optional['SingleQubitCliffordGate']
</code></pre>

Creates Clifford gate with given unitary (up to global phase).


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`u`
</td>
<td>
2x2 unitary matrix of a Clifford gate.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
SingleQubitCliffordGate, whose matrix is equal to given matrix (up
to global phase), or `None` if `u` is not a matrix of a single-qubit
Clifford gate.
</td>
</tr>

</table>



<h3 id="from_xz_map"><code>from_xz_map</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_xz_map(
    x_to: Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool],
    z_to: Tuple[<a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>, bool]
) -> "SingleQubitCliffordGate"
</code></pre>

Returns a SingleQubitCliffordGate for the specified transforms.
The Y transform is derived from the X and Z.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x_to`
</td>
<td>
Which Pauli to transform X to and if it should negate.
</td>
</tr><tr>
<td>
`z_to`
</td>
<td>
Which Pauli to transform Z to and if it should negate.
</td>
</tr>
</table>



<h3 id="merged_with"><code>merged_with</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>merged_with(
    second: "SingleQubitCliffordGate"
) -> "SingleQubitCliffordGate"
</code></pre>

Returns a SingleQubitCliffordGate such that the circuits
    --output-- and --self--second--
are equivalent up to global phase.

<h3 id="num_qubits"><code>num_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_qubits() -> int
</code></pre>

The number of qubits this gate acts on.


<h3 id="on"><code>on</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on(
    *qubits
) -> "Operation"
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



<h3 id="on_each"><code>on_each</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/gate_features.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_each(
    *targets
) -> List[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]
</code></pre>

Returns a list of operations applying the gate to all targets.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`*targets`
</td>
<td>
The qubits to apply this gate to.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Operations applying this gate to the target qubits.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>
<tr class="alt">
<td colspan="2">
ValueError if targets are not instances of Qid or List[Qid].
</td>
</tr>

</table>



<h3 id="transform"><code>transform</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>transform(
    pauli: <a href="../../cirq/ops/Pauli.md"><code>cirq.ops.Pauli</code></a>
) -> <a href="../../cirq/ops/PauliTransform.md"><code>cirq.ops.PauliTransform</code></a>
</code></pre>




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




<h3 id="__mul__"><code>__mul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    other: Union[complex, float, int]
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__neg__"><code>__neg__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__() -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/clifford_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    exponent
) -> "SingleQubitCliffordGate"
</code></pre>




<h3 id="__rmul__"><code>__rmul__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    other: Union[complex, float, int]
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__sub__"><code>__sub__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    other: Union['Gate', 'cirq.LinearCombinationOfGates']
) -> "cirq.LinearCombinationOfGates"
</code></pre>




<h3 id="__truediv__"><code>__truediv__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    other: Union[complex, float, int]
) -> "cirq.LinearCombinationOfGates"
</code></pre>






