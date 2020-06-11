<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.quil" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.quil

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/quil.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the QUIL code for the given value.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.quil`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.quil(
    val: Any,
    *,
    qubits: Optional[Iterable['cirq.Qid']] = None,
    formatter: Optional[<a href="../../cirq/protocols/QuilFormatter.md"><code>cirq.protocols.QuilFormatter</code></a>] = None
)
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
The value to turn into QUIL code.
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
`formatter`
</td>
<td>
A `QuilFormatter` object for properly ouputting the `_quil_`
method in a QUIL format.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The result of `val._quil_(...) if `val` has a `_quil_` method.
Otherwise, returns `None`. (`None` normally indicates that the
`_decompose_` function should be called on `val`)
</td>
</tr>

</table>

