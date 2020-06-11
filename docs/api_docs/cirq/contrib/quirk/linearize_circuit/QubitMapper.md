<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.quirk.linearize_circuit.QubitMapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="map_moment"/>
<meta itemprop="property" content="map_operation"/>
<meta itemprop="property" content="optimize_circuit"/>
</div>

# cirq.contrib.quirk.linearize_circuit.QubitMapper

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/quirk/linearize_circuit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>





<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.quirk.linearize_circuit.QubitMapper(
    qubit_map: Callable[[<a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>], <a href="../../../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>]
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="map_moment"><code>map_moment</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/quirk/linearize_circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map_moment(
    moment: <a href="../../../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>
) -> <a href="../../../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>
</code></pre>




<h3 id="map_operation"><code>map_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/quirk/linearize_circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map_operation(
    operation: <a href="../../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
</code></pre>




<h3 id="optimize_circuit"><code>optimize_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/quirk/linearize_circuit.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>optimize_circuit(
    circuit: <a href="../../../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
)
</code></pre>






