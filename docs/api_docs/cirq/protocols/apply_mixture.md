<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.apply_mixture" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.apply_mixture

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/apply_mixture_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



High performance evolution under a mixture of unitaries evolution.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.apply_mixture`, `cirq.protocols.apply_mixture_protocol.apply_mixture`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.apply_mixture(
    val: Any,
    args: <a href="../../cirq/protocols/ApplyMixtureArgs.md"><code>cirq.protocols.ApplyMixtureArgs</code></a>,
    *,
    default: <a href="../../cirq/protocols/apply_mixture_protocol/TDefault.md"><code>cirq.protocols.apply_mixture_protocol.TDefault</code></a> = cirq.protocols.apply_mixture_protocol.RaiseTypeErrorIfNotProvided
) -> Union[np.ndarray, <a href="../../cirq/protocols/apply_mixture_protocol/TDefault.md"><code>cirq.protocols.apply_mixture_protocol.TDefault</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

Follows the steps below to attempt to apply a mixture:

A. Try to use `val._apply_mixture_(args)`.
    1. If `_apply_mixture_` is not present or returns NotImplemented
        go to step B.
    2. If '_apply_mixture_' is present and returns None conclude that
        `val` has no effect and return.
    3. If '_apply_mixture_' is present and returns a numpy array conclude
        that the mixture was applied successfully and forward result to
        caller.

B. Construct an ApplyUnitaryArgs object `uargs` from `args` and then
    try to use <a href="../../cirq/protocols/apply_unitary.md"><code>cirq.apply_unitary(val, uargs, None)</code></a>.
    1. If `None` is returned then go to step C.
    2. If a numpy array is returned forward this result back to the caller
        and return.

C. Try to use `val._mixture_()`.
    1. If '_mixture_' is not present or returns NotImplemented
        go to step D.
    2. If '_mixture_' is present and returns None conclude that `val` has
        no effect and return.
    3. If '_mixture_' returns a list of tuples, loop over the list and
        examine each tuple. If the tuple is of the form
        `(probability, np.ndarray)` use matrix multiplication to apply it.
        If the tuple is of the form `(probability, op)` where op is any op,
        attempt to use <a href="../../cirq/protocols/apply_unitary.md"><code>cirq.apply_unitary(op, uargs, None)</code></a>. If this
        operation returns None go to step D. Otherwise return the resulting
        state after all of the tuples have been applied.

D. Raise TypeError or return `default`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value with a mixture to apply to the target.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
A mutable <a href="../../cirq/protocols/ApplyMixtureArgs.md"><code>cirq.ApplyMixtureArgs</code></a> object describing the target
tensor, available workspace, and left and right axes to operate on.
The attributes of this object will be mutated as part of computing
the result.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
What should be returned if `val` doesn't have a mixture. If
not specified, a TypeError is raised instead of returning a default
value.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If the receiving object is not able to apply a mixture,
the specified default value is returned (or a TypeError is raised). If
this occurs, then `target_tensor` should not have been mutated.

If the receiving object was able to work inline, directly
mutating `target_tensor` it will return `target_tensor`. The caller is
responsible for checking if the result is `target_tensor`.

If the receiving object wrote its output over `out_buffer`, the
result will be `out_buffer`. The caller is responsible for
checking if the result is `out_buffer` (and e.g. swapping
the buffer for the target tensor before the next call).

Note that it is an error for the return object to be either of the
auxiliary buffers, and the method will raise an AssertionError if
this contract is violated.

The receiving object may also write its output over a new buffer
that it created, in which case that new array is returned.
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
`val` doesn't have a mixture and `default` wasn't specified.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
Different left and right shapes of `args.target_tensor`
selected by `left_axes` and `right_axes` or `qid_shape(val)` doesn't
equal the left and right shapes.
</td>
</tr><tr>
<td>
`AssertionError`
</td>
<td>
`_apply_mixture_` returned an auxiliary buffer.
</td>
</tr>
</table>

