<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.circuit_to_latex_using_qcircuit" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.circuit_to_latex_using_qcircuit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/qcircuit/qcircuit_diagram.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a QCircuit-based latex diagram of the given circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.qcircuit.circuit_to_latex_using_qcircuit`, `cirq.contrib.qcircuit.qcircuit_diagram.circuit_to_latex_using_qcircuit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.circuit_to_latex_using_qcircuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    qubit_order: <a href="../../cirq/ops/QubitOrderOrList.md"><code>cirq.ops.QubitOrderOrList</code></a> = cirq.ops.QubitOrder.DEFAULT
) -> str
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit to represent in latex.
</td>
</tr><tr>
<td>
`qubit_order`
</td>
<td>
Determines the order of qubit wires in the diagram.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Latex code for the diagram.
</td>
</tr>

</table>

