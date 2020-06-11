<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.qasm" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.qasm

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/qasm.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns QASM code for the given value, if possible.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.qasm`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.qasm(
    val: Any,
    *,
    args: Optional[<a href="../../cirq/protocols/QasmArgs.md"><code>cirq.protocols.QasmArgs</code></a>] = None,
    qubits: Optional[Iterable['cirq.Qid']] = None,
    default: TDefault = RaiseTypeErrorIfNotProvided
) -> Union[str, TDefault]
</code></pre>



<!-- Placeholder for "Used in" -->

Different values require different sets of arguments. The general rule of
thumb is that circuits don't need any, operations need a `QasmArgs`, and
gates need both a `QasmArgs` and `qubits`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value to turn into QASM code.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
A `QasmArgs` object to pass into the value's `_qasm_` method.
This is for needed for objects that only have a local idea of what's
going on, e.g. a <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a> in a bigger <a href="../../cirq/circuits/Circuit.md"><code>cirq.Circuit</code></a>
involving qubits that the operation wouldn't otherwise know about.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
A list of qubits that the value is being applied to. This is
needed for <a href="../../cirq/ops/Gate.md"><code>cirq.Gate</code></a> values, which otherwise wouldn't know what
qubits to talk about.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
A default result to use if the value doesn't have a
`_qasm_` method or that method returns `NotImplemented` or
`None`. If not specified, undecomposable values cause a `TypeError`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The result of `val._qasm_(...)`, if `val` has a `_qasm_`
method and it didn't return `NotImplemented` or `None`. Otherwise
`default` is returned, if it was specified. Otherwise an error is
raised.
</td>
</tr>

</table>



#### TypeError:

`val` didn't have a `_qasm_` method (or that method returned
`NotImplemented` or `None`) and `default` wasn't set.
