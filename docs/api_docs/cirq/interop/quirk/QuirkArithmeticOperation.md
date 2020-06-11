<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.interop.quirk.QuirkArithmeticOperation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="controlled_by"/>
<meta itemprop="property" content="registers"/>
<meta itemprop="property" content="transform_qubits"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="with_probability"/>
<meta itemprop="property" content="with_qubits"/>
<meta itemprop="property" content="with_registers"/>
<meta itemprop="property" content="with_tags"/>
</div>

# cirq.interop.quirk.QuirkArithmeticOperation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/arithmetic_cells.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Applies arithmetic to a target and some inputs.

Inherits From: [`ArithmeticOperation`](../../../cirq/ops/ArithmeticOperation.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.interop.quirk.cells.QuirkArithmeticOperation`, `cirq.interop.quirk.cells.arithmetic_cells.QuirkArithmeticOperation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.interop.quirk.QuirkArithmeticOperation(
    identifier: str,
    target: Sequence['cirq.Qid'],
    inputs: Sequence[Union[Sequence['cirq.Qid'], int]]
)
</code></pre>



<!-- Placeholder for "Used in" -->

Implements Quirk-specific implicit effects like assuming that the presence
of an 'r' input implies modular arithmetic.

In Quirk, modular operations have no effect on values larger than the
modulus. This convention is used because unitarity forces *some* convention
on out-of-range values (they cannot simply disappear or raise exceptions),
and the simplest is to do nothing. This call handles ensuring that happens,
and ensuring the new target register value is normalized modulo the modulus.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`gate`
</td>
<td>

</td>
</tr><tr>
<td>
`operation`
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

<h3 id="apply"><code>apply</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/arithmetic_cells.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>apply(
    *registers
) -> Union[int, Iterable[int]]
</code></pre>

Returns the result of the operation operating on classical values.

For example, an addition takes two values (the target and the source),
adds the source into the target, then returns the target and source
as the new register values.

The `apply` method is permitted to be sloppy in three ways:

1. The `apply` method is permitted to return values that have more bits
    than the registers they will be stored into. The extra bits are
    simply dropped. For example, if the value 5 is returned for a 2
    qubit register then 5 % 2**2 = 1 will be used instead. Negative
    values are also permitted. For example, for a 3 qubit register the
    value -2 becomes -2 % 2**3 = 6.
2. When the value of the last `k` registers is not changed by the
    operation, the `apply` method is permitted to omit these values
    from the result. That is to say, when the length of the output is
    less than the length of the input, it is padded up to the intended
    length by copying from the same position in the input.
3. When only the first register's value changes, the `apply` method is
    permitted to return an `int` instead of a sequence of ints.

The `apply` method *must* be reversible. Otherwise the operation will
not be unitary, and incorrect behavior will result.

#### Examples:


A fully detailed adder:

```
def apply(self, target, offset):
    return (target + offset) % 2**len(self.target_register), offset
```

The same adder, with less boilerplate due to the details being
handled by the `ArithmeticOperation` class:

```
def apply(self, target, offset):
    return target + offset
```


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



<h3 id="registers"><code>registers</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/arithmetic_cells.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>registers() -> Sequence[Union[int, Sequence['cirq.Qid']]]
</code></pre>

The data acted upon by the arithmetic operation.

Each register in the list can either be a classical constant (an `int`),
or else a list of qubits/qudits (a `List[cirq.Qid]`). Registers that
are set to a classical constant must not be mutated by the arithmetic
operation (their value must remain fixed when passed to `apply`).

Registers are big endian. The first qubit is the most significant, the
last qubit is the 1s qubit, the before last qubit is the 2s qubit, etc.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A list of constants and qubit groups that the operation will act
upon.
</td>
</tr>

</table>



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



<h3 id="with_probability"><code>with_probability</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/raw_types.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_probability(
    probability: "cirq.TParamVal"
) -> "cirq.Operation"
</code></pre>




<h3 id="with_qubits"><code>with_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/arithmetic_operation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_qubits(
    *new_qubits
) -> "cirq.Operation"
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



<h3 id="with_registers"><code>with_registers</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/cells/arithmetic_cells.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_registers(
    *new_registers
) -> "QuirkArithmeticOperation"
</code></pre>

Returns the same operation targeting different registers.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`new_registers`
</td>
<td>
The new values that should be returned by the
`registers` method.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance of the same kind of operation, but acting on different
registers.
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






