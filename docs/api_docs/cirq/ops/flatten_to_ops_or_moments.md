<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.flatten_to_ops_or_moments" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.flatten_to_ops_or_moments

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/op_tree.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Performs an in-order iteration OP_TREE, yielding ops and moments.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.flatten_to_ops_or_moments`, `cirq.ops.op_tree.flatten_to_ops_or_moments`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.flatten_to_ops_or_moments(
    root: <a href="../../cirq/ops/OP_TREE.md"><code>cirq.ops.OP_TREE</code></a>
) -> Iterator[Union[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>, <a href="../../cirq/ops/Moment.md"><code>cirq.ops.Moment</code></a>]]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`root`
</td>
<td>
The operation or tree of operations to iterate.
</td>
</tr>
</table>



#### Yields:

Operations or moments from the tree.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
root isn't a valid OP_TREE.
</td>
</tr>
</table>

