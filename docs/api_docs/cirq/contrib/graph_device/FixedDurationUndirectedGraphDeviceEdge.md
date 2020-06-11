<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.graph_device.FixedDurationUndirectedGraphDeviceEdge" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="duration_of"/>
<meta itemprop="property" content="validate_operation"/>
</div>

# cirq.contrib.graph_device.FixedDurationUndirectedGraphDeviceEdge

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An edge of an undirected graph device on which every operation is

Inherits From: [`UndirectedGraphDeviceEdge`](../../../cirq/contrib/graph_device/graph_device/UndirectedGraphDeviceEdge.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.contrib.graph_device.graph_device.FixedDurationUndirectedGraphDeviceEdge`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.graph_device.FixedDurationUndirectedGraphDeviceEdge(
    duration: <a href="../../../cirq/value/Duration.md"><code>cirq.value.Duration</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
allowed and has the same duration.

## Methods

<h3 id="duration_of"><code>duration_of</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>duration_of(
    operation: <a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> <a href="../../../cirq/value/Duration.md"><code>cirq.value.Duration</code></a>
</code></pre>




<h3 id="validate_operation"><code>validate_operation</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/graph_device/graph_device.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_operation(
    operation: <a href="../../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
) -> None
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






