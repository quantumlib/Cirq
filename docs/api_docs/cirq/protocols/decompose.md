<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.decompose" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.decompose

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/decompose_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Recursively decomposes a value into <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a>s meeting a criteria.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.decompose`, `cirq.protocols.decompose_protocol.decompose`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.decompose(
    val: <a href="../../cirq/protocols/decompose_protocol/TValue.md"><code>cirq.protocols.decompose_protocol.TValue</code></a>,
    *,
    intercepting_decomposer: Optional[<a href="../../cirq/protocols/decompose_protocol/OpDecomposer.md"><code>cirq.protocols.decompose_protocol.OpDecomposer</code></a>] = None,
    fallback_decomposer: Optional[<a href="../../cirq/protocols/decompose_protocol/OpDecomposer.md"><code>cirq.protocols.decompose_protocol.OpDecomposer</code></a>] = None,
    keep: Optional[Callable[['cirq.Operation'], bool]] = None,
    on_stuck_raise: Union[None, Exception, Callable[['cirq.Operation'], Union[None, Exception]]] = _value_error_describing_bad_operation
) -> List['cirq.Operation']
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value to decompose into operations.
</td>
</tr><tr>
<td>
`intercepting_decomposer`
</td>
<td>
An optional method that is called before the
default decomposer (the value's `_decompose_` method). If
`intercepting_decomposer` is specified and returns a result that
isn't `NotImplemented` or `None`, that result is used. Otherwise the
decomposition falls back to the default decomposer.

Note that `val` will be passed into `intercepting_decomposer`, even
if `val` isn't a <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a>.
</td>
</tr><tr>
<td>
`fallback_decomposer`
</td>
<td>
An optional decomposition that used after the
`intercepting_decomposer` and the default decomposer (the value's
`_decompose_` method) both fail.
</td>
</tr><tr>
<td>
`keep`
</td>
<td>
A predicate that determines if the initial operation or
intermediate decomposed operations should be kept or else need to be
decomposed further. If `keep` isn't specified, it defaults to "value
can't be decomposed anymore".
</td>
</tr><tr>
<td>
`on_stuck_raise`
</td>
<td>
If there is an operation that can't be decomposed and
also can't be kept, `on_stuck_raise` is used to determine what error
to raise. `on_stuck_raise` can either directly be an `Exception`, or
a method that takes the problematic operation and returns an
`Exception`. If `on_stuck_raise` is set to `None` or a method that
returns `None`, undecomposable operations are simply silently kept.
`on_stuck_raise` defaults to a `ValueError` describing the unwanted
undecomposable operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of operations that the given value was decomposed into. If
`on_stuck_raise` isn't set to None, all operations in the list will
satisfy the predicate specified by `keep`.
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
`val` isn't a <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a> and can't be decomposed even once.
(So it's not possible to return a list of operations.)
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
Default type of error raised if there's an undecomposable operation
that doesn't satisfy the given `keep` predicate.
</td>
</tr><tr>
<td>
`TError`
</td>
<td>
Custom type of error raised if there's an undecomposable operation
that doesn't satisfy the given `keep` predicate.
</td>
</tr>
</table>

