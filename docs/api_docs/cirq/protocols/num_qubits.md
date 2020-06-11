<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.num_qubits" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.num_qubits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/qid_shape_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the number of qubits, qudits, or qids `val` operates on.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.num_qubits`, `cirq.protocols.qid_shape_protocol.num_qubits`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.num_qubits(
    val: Any,
    default: <a href="../../cirq/protocols/qid_shape_protocol/TDefault.md"><code>cirq.protocols.qid_shape_protocol.TDefault</code></a> = RaiseTypeErrorIfNotProvidedInt
) -> Union[int, <a href="../../cirq/protocols/qid_shape_protocol/TDefault.md"><code>cirq.protocols.qid_shape_protocol.TDefault</code></a>]
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
The value to get the number of qubits from.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Determines the fallback behavior when `val` doesn't have
a number of qubits. If `default` is not set, a TypeError is raised.
If default is set to a value, that value is returned.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has a `_num_qubits_` method and its result is not
NotImplemented, that result is returned. Otherwise, if `val` has a
`_qid_shape_` method, the number of qubits is computed from the length
of the shape and returned e.g. `len(shape)`. If neither method returns a
value other than NotImplemented and a default value was specified, the
default value is returned.
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
`val` doesn't have either a `_num_qubits_` or a `_qid_shape_`
method (or they returned NotImplemented) and also no default value
was specified.
</td>
</tr>
</table>

