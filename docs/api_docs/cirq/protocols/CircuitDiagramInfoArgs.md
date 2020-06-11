<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.CircuitDiagramInfoArgs" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="format_radians"/>
<meta itemprop="property" content="format_real"/>
<meta itemprop="property" content="with_args"/>
<meta itemprop="property" content="UNINFORMED_DEFAULT"/>
</div>

# cirq.protocols.CircuitDiagramInfoArgs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/circuit_diagram_info_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A request for information on drawing an operation in a circuit diagram.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CircuitDiagramInfoArgs`, `cirq.protocols.circuit_diagram_info_protocol.CircuitDiagramInfoArgs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.CircuitDiagramInfoArgs(
    known_qubits: Optional[Iterable['cirq.Qid']],
    known_qubit_count: Optional[int],
    use_unicode_characters: bool,
    precision: Optional[int],
    qubit_map: Optional[Dict['cirq.Qid', int]],
    include_tags: bool = True
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`known_qubits`
</td>
<td>
The qubits the gate is being applied to. None means this
information is not known by the caller.
</td>
</tr><tr>
<td>
`known_qubit_count`
</td>
<td>
The number of qubits the gate is being applied to
None means this information is not known by the caller.
</td>
</tr><tr>
<td>
`use_unicode_characters`
</td>
<td>
If true, the wire symbols are permitted to
include unicode characters (as long as they work well in fixed
width fonts). If false, use only ascii characters. ASCII is
preferred in cases where UTF8 support is done poorly, or where
the fixed-width font being used to show the diagrams does not
properly handle unicode characters.
</td>
</tr><tr>
<td>
`precision`
</td>
<td>
The number of digits after the decimal to show for numbers in
the text diagram. None means use full precision.
</td>
</tr><tr>
<td>
`qubit_map`
</td>
<td>
The map from qubits to diagram positions.
</td>
</tr><tr>
<td>
`include_tags`
</td>
<td>
Whether to print tags from TaggedOperations
</td>
</tr>
</table>



## Methods

<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/circuit_diagram_info_protocol.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy()
</code></pre>




<h3 id="format_radians"><code>format_radians</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/circuit_diagram_info_protocol.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>format_radians(
    radians: Union[sympy.Basic, int, float]
) -> str
</code></pre>

Returns angle in radians as a human-readable string.


<h3 id="format_real"><code>format_real</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/circuit_diagram_info_protocol.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>format_real(
    val: Union[sympy.Basic, int, float]
) -> str
</code></pre>




<h3 id="with_args"><code>with_args</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/circuit_diagram_info_protocol.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_args(
    **kwargs
)
</code></pre>




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






## Class Variables

* `UNINFORMED_DEFAULT` <a id="UNINFORMED_DEFAULT"></a>
