<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.CircuitDiagramInfo" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="with_wire_symbols"/>
</div>

# cirq.protocols.CircuitDiagramInfo

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/circuit_diagram_info_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Describes how to draw an operation in a circuit diagram.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CircuitDiagramInfo`, `cirq.protocols.circuit_diagram_info_protocol.CircuitDiagramInfo`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.CircuitDiagramInfo(
    wire_symbols: Iterable[str],
    exponent: Any = 1,
    connected: bool = True,
    exponent_qubit_index: Optional[int] = None,
    auto_exponent_parens: bool = True
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="with_wire_symbols"><code>with_wire_symbols</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/circuit_diagram_info_protocol.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>with_wire_symbols(
    new_wire_symbols: Iterable[str]
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






