<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.executor.ExecutionStrategy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="get_operations"/>
<meta itemprop="property" content="keep_acquaintance"/>
</div>

# cirq.contrib.acquaintance.executor.ExecutionStrategy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/executor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tells StrategyExecutor how to execute an acquaintance strategy.

<!-- Placeholder for "Used in" -->

An execution strategy tells StrategyExecutor how to execute an
acquaintance strategy, i.e. what gates to implement at the available
acquaintance opportunities.



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

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/executor.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
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
