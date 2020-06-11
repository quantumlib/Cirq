<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.decompose_once_with_qubits" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.decompose_once_with_qubits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/decompose_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decomposes a value into operations on the given qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.decompose_once_with_qubits`, `cirq.protocols.decompose_protocol.decompose_once_with_qubits`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.decompose_once_with_qubits(
    val: Any,
    qubits: Iterable['cirq.Qid'],
    default=cirq.protocols.decompose_protocol.RaiseTypeErrorIfNotProvided
)
</code></pre>



<!-- Placeholder for "Used in" -->

This method is used when decomposing gates, which don't know which qubits
they are being applied to unless told. It decomposes the gate exactly once,
instead of decomposing it and then continuing to decomposing the decomposed
operations recursively until some criteria is met.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value to call `._decompose_(qubits)` on, if possible.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
The value to pass into the named `qubits` parameter of
`val._decompose_`.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
A default result to use if the value doesn't have a
`_decompose_` method or that method returns `NotImplemented` or
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
The result of `val._decompose_(qubits)`, if `val` has a
`_decompose_` method and it didn't return `NotImplemented` or `None`.
Otherwise `default` is returned, if it was specified. Otherwise an error
is raised.
</td>
</tr>

</table>



#### TypeError:

`val` didn't have a `_decompose_` method (or that method returned
`NotImplemented` or `None`) and `default` wasn't set.
