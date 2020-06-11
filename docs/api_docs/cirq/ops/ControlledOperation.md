<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.ControlledOperation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="controlled_by"/>
<meta itemprop="property" content="transform_qubits"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="with_probability"/>
<meta itemprop="property" content="with_qubits"/>
<meta itemprop="property" content="with_tags"/>
</div>

# cirq.ops.ControlledOperation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/controlled_operation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Augments existing operations to have one or more control qubits.

Inherits From: [`Operation`](../../cirq/ops/Operation.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ControlledOperation`, `cirq.ops.controlled_operation.ControlledOperation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.ControlledOperation(
    controls: Sequence['cirq.Qid'],
    sub_operation: "cirq.Operation",
    control_values: Optional[Sequence[Union[int, Collection[int]]]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This object is typically created via `operation.controlled_by(*qubits)`.



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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/controlled_operation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_qubits(
    *new_qubits
)
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




<h3 id="__pow__"><code>__pow__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/controlled_operation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    exponent: Any
) -> "ControlledOperation"
</code></pre>






