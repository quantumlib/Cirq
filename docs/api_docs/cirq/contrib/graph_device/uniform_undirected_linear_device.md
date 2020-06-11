<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.graph_device.uniform_undirected_linear_device" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.contrib.graph_device.uniform_undirected_linear_device

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/uniform_graph_device.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A uniform , undirected graph device whose qubits are arranged

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.graph_device.uniform_graph_device.uniform_undirected_linear_device`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.graph_device.uniform_undirected_linear_device(
    n_qubits: int,
    edge_labels: Mapping[int, Optional[<a href="../../../cirq/contrib/graph_device/graph_device/UndirectedGraphDeviceEdge.md"><code>cirq.contrib.graph_device.graph_device.UndirectedGraphDeviceEdge</code></a>]]
) -> <a href="../../../cirq/contrib/graph_device/UndirectedGraphDevice.md"><code>cirq.contrib.graph_device.UndirectedGraphDevice</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
on a line.

Uniformity refers to the fact that all edges of the same size have the same
label.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`n_qubits`
</td>
<td>
The number of qubits.
</td>
</tr><tr>
<td>
`edge_labels`
</td>
<td>
The labels to apply to all edges of a given size.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
keys to edge_labels are not all at least 1.
</td>
</tr>
</table>

