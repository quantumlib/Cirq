<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.apply_unitaries" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.apply_unitaries

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_unitary_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Apply a series of unitaries onto a state tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.apply_unitaries`, `cirq.protocols.apply_unitary_protocol.apply_unitaries`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.apply_unitaries(
    unitary_values: Iterable[Any],
    qubits: Sequence['cirq.Qid'],
    args: Optional[<a href="../../cirq/protocols/ApplyUnitaryArgs.md"><code>cirq.protocols.ApplyUnitaryArgs</code></a>] = None,
    default: Any = cirq.protocols.apply_unitary_protocol.RaiseTypeErrorIfNotProvided
) -> Optional[np.ndarray]
</code></pre>



<!-- Placeholder for "Used in" -->

Uses <a href="../../cirq/protocols/apply_unitary.md"><code>cirq.apply_unitary</code></a> on each of the unitary values, to apply them to
the state tensor from the `args` argument.

CAUTION: if one of the given unitary values does not have a unitary effect,
forcing the method to terminate, the method will not rollback changes
from previous unitary values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`unitary_values`
</td>
<td>
The values with unitary effects to apply to the target.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
The qubits that will be targeted by the unitary values. These
qubits match up, index by index, with the `indices` property of the
`args` argument.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
A mutable <a href="../../cirq/protocols/ApplyUnitaryArgs.md"><code>cirq.ApplyUnitaryArgs</code></a> object describing the target
tensor, available workspace, and axes to operate on. The attributes
of this object will be mutated as part of computing the result. If
not specified, this defaults to the zero state of the given qubits
with an axis ordering matching the given qubit ordering.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
What should be returned if any of the unitary values actually
don't have a unitary effect. If not specified, a TypeError is
raised instead of returning a default value.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If any of the unitary values do not have a unitary effect, the
specified default value is returned (or a TypeError is raised).
</td>
</tr>
<tr>
<td>
`CAUTION`
</td>
<td>
If this occurs, the contents of `args.target_tensor`
and `args.available_buffer` may have been mutated.

If all of the unitary values had a unitary effect that was
successfully applied, this method returns the `np.ndarray`
storing the final result. This `np.ndarray` may be
`args.target_tensor`, `args.available_buffer`, or some
other instance. The caller is responsible for dealing with
this potential aliasing of the inputs and the result.
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
An item from `unitary_values` doesn't have a unitary effect
and `default` wasn't specified.
</td>
</tr>
</table>

