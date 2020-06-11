<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.measure_each" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.measure_each

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/measure_util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a list of operations individually measuring the given qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.measure_each`, `cirq.ops.measure_util.measure_each`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.measure_each(
    *qubits,
    key_func: Callable[[<a href="../../cirq/ops/Qid.md"><code>cirq.ops.Qid</code></a>], str] = str
) -> List[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

The qubits are measured in the computational basis.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`*qubits`
</td>
<td>
The qubits to measure.
</td>
</tr><tr>
<td>
`key_func`
</td>
<td>
Determines the key of the measurements of each qubit. Takes
the qubit and returns the key for that qubit. Defaults to str.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of operations individually measuring the given qubits.
</td>
</tr>

</table>

