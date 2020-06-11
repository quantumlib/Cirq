<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.apply_unitary" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.apply_unitary

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_unitary_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



High performance left-multiplication of a unitary effect onto a tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.apply_unitary`, `cirq.protocols.apply_unitary_protocol.apply_unitary`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.apply_unitary(
    unitary_value: Any,
    args: <a href="../../cirq/protocols/ApplyUnitaryArgs.md"><code>cirq.protocols.ApplyUnitaryArgs</code></a>,
    default: <a href="../../cirq/protocols/apply_unitary_protocol/TDefault.md"><code>cirq.protocols.apply_unitary_protocol.TDefault</code></a> = cirq.protocols.apply_unitary_protocol.RaiseTypeErrorIfNotProvided,
    *,
    allow_decompose: bool = True
) -> Union[np.ndarray, <a href="../../cirq/protocols/apply_unitary_protocol/TDefault.md"><code>cirq.protocols.apply_unitary_protocol.TDefault</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

Applies the unitary effect of `unitary_value` to the tensor specified in
`args` by using the following strategies:

A. Try to use `unitary_value._apply_unitary_(args)`.
    Case a) Method not present or returns `NotImplemented`.
        Continue to next strategy.
    Case b) Method returns `None`.
        Conclude `unitary_value` has no unitary effect.
    Case c) Method returns a numpy array.
        Forward the successful result to the caller.

B. Try to use `unitary_value._unitary_()`.
    Case a) Method not present or returns `NotImplemented`.
        Continue to next strategy.
    Case b) Method returns `None`.
        Conclude `unitary_value` has no unitary effect.
    Case c) Method returns a numpy array.
        Multiply the matrix onto the target tensor and return to the caller.

C. Try to use `unitary_value._decompose_()` (if `allow_decompose`).
    Case a) Method not present or returns `NotImplemented` or `None`.
        Continue to next strategy.
    Case b) Method returns an OP_TREE.
        Delegate to <a href="../../cirq/protocols/apply_unitaries.md"><code>cirq.apply_unitaries</code></a>.

D. Conclude that `unitary_value` has no unitary effect.

The order that the strategies are tried depends on the number of qubits
being operated on. For small numbers of qubits (4 or less) the order is
ABCD. For larger numbers of qubits the order is ACBD (because it is expected
that decomposing will outperform generating the raw matrix).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`unitary_value`
</td>
<td>
The value with a unitary effect to apply to the target.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
A mutable <a href="../../cirq/protocols/ApplyUnitaryArgs.md"><code>cirq.ApplyUnitaryArgs</code></a> object describing the target
tensor, available workspace, and axes to operate on. The attributes
of this object will be mutated as part of computing the result.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
What should be returned if `unitary_value` doesn't have a
unitary effect. If not specified, a TypeError is raised instead of
returning a default value.
</td>
</tr><tr>
<td>
`allow_decompose`
</td>
<td>
Defaults to True. If set to False, and applying the
unitary effect requires decomposing the object, the method will
pretend the object has no unitary effect.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If the receiving object does not have a unitary effect, then the
specified default value is returned (or a TypeError is raised). If
this occurs, then `target_tensor` should not have been mutated.

Otherwise the result is the `np.ndarray` instance storing the result.
This may be `args.target_tensor`, `args.available_workspace`, or some
other numpy array. It is the caller's responsibility to correctly handle
all three of these cases. In all cases `args.target_tensor` and
`args.available_buffer` may have been mutated.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
`unitary_value` doesn't have a unitary effect and `default`
wasn't specified.
</td>
</tr>
</table>

