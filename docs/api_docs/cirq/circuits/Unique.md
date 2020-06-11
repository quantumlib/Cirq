<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.Unique" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__new__"/>
</div>

# cirq.circuits.Unique

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A wrapper for a value that doesn't compare equal to other instances.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.Unique`, `cirq.circuits.circuit_dag.Unique`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.circuits.Unique(
    val: <a href="../../cirq/circuits/circuit_dag/T.md"><code>cirq.circuits.circuit_dag.T</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

For example: 5 == 5 but Unique(5) != Unique(5).

Unique is used by CircuitDag to wrap operations because nodes in a graph
are considered the same node if they compare equal to each other.  X(q0)
in one moment of a Circuit and X(q0) in another moment of the Circuit are
wrapped by Unique(X(q0)) so they are distinct nodes in the graph.

## Methods

<h3 id="__ge__"><code>__ge__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    other, NotImplemented=NotImplemented
)
</code></pre>

Return a >= b.  Computed by @total_ordering from (not a < b).


<h3 id="__gt__"><code>__gt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    other, NotImplemented=NotImplemented
)
</code></pre>

Return a > b.  Computed by @total_ordering from (not a < b) and (a != b).


<h3 id="__le__"><code>__le__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    other, NotImplemented=NotImplemented
)
</code></pre>

Return a <= b.  Computed by @total_ordering from (a < b) or (a == b).


<h3 id="__lt__"><code>__lt__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    other
)
</code></pre>

Return self<value.




