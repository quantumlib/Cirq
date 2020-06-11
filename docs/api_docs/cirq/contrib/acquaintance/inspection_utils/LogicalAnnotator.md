<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.inspection_utils.LogicalAnnotator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_operations"/>
<meta itemprop="property" content="keep_acquaintance"/>
</div>

# cirq.contrib.acquaintance.inspection_utils.LogicalAnnotator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/inspection_utils.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Realizes acquaintance opportunities.

Inherits From: [`ExecutionStrategy`](../../../../cirq/contrib/acquaintance/executor/ExecutionStrategy.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.acquaintance.inspection_utils.LogicalAnnotator(
    initial_mapping: <a href="../../../../cirq/contrib/acquaintance/executor/LogicalMapping.md"><code>cirq.contrib.acquaintance.executor.LogicalMapping</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
    



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`device`
</td>
<td>
The device for which the executed acquaintance strategy should be
valid.
</td>
</tr><tr>
<td>
`initial_mapping`
</td>
<td>
The initial mapping of logical indices to qubits.
</td>
</tr>
</table>



## Methods

<h3 id="get_operations"><code>get_operations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/inspection_utils.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_operations(
    indices: Sequence[<a href="../../../../cirq/contrib/acquaintance/executor/LogicalIndex.md"><code>cirq.contrib.acquaintance.executor.LogicalIndex</code></a>],
    qubits: Sequence['cirq.Qid']
) -> "cirq.OP_TREE"
</code></pre>

Gets the logical operations to apply to qubits.


<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/executor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>

Call self as a function.




## Class Variables

* `keep_acquaintance = False` <a id="keep_acquaintance"></a>
