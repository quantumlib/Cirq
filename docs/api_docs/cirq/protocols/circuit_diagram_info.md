<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.circuit_diagram_info" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.circuit_diagram_info

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/circuit_diagram_info_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Requests information on drawing an operation in a circuit diagram.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.circuit_diagram_info`, `cirq.protocols.circuit_diagram_info_protocol.circuit_diagram_info`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.circuit_diagram_info(
    val: Any,
    args: Optional[<a href="../../cirq/protocols/CircuitDiagramInfoArgs.md"><code>cirq.protocols.CircuitDiagramInfoArgs</code></a>] = None,
    default=cirq.protocols.circuit_diagram_info_protocol.RaiseTypeErrorIfNotProvided
)
</code></pre>



<!-- Placeholder for "Used in" -->

Calls _circuit_diagram_info_ on `val`. If `val` doesn't have
_circuit_diagram_info_, or it returns NotImplemented, that indicates that
diagram information is not available.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The operation or gate that will need to be drawn.
</td>
</tr><tr>
<td>
`args`
</td>
<td>
A CircuitDiagramInfoArgs describing the desired drawing style.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
A default result to return if the value doesn't have circuit
diagram information. If not specified, a TypeError is raised
instead.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has no _circuit_diagram_info_ method or it returns
NotImplemented, then `default` is returned (or a TypeError is
raised if no `default` is specified).

Otherwise, the value returned by _circuit_diagram_info_ is returned.
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
`val` doesn't have circuit diagram information and `default` was
not specified.
</td>
</tr>
</table>

