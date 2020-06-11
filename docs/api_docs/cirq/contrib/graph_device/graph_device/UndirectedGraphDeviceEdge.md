<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.graph_device.graph_device.UndirectedGraphDeviceEdge" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="duration_of"/>
<meta itemprop="property" content="validate_operation"/>
</div>

# cirq.contrib.graph_device.graph_device.UndirectedGraphDeviceEdge

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An edge of an undirected graph device.

<!-- Placeholder for "Used in" -->
    

## Methods

<h3 id="duration_of"><code>duration_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>duration_of(
    operation: <a href="../../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../../../cirq/value/Duration.md"><code>cirq.value.Duration</code></a>
</code></pre>




<h3 id="validate_operation"><code>validate_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>validate_operation(
    operation: <a href="../../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> None
</code></pre>






