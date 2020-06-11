<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.measure" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.measure

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/measure_util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a single MeasurementGate applied to all the given qubits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.measure`, `cirq.ops.measure_util.measure`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.measure(
    *target,
    key: Optional[str] = None,
    invert_mask: Tuple[bool, ...] = ()
) -> <a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

The qubits are measured in the computational basis.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`*target`
</td>
<td>
The qubits that the measurement gate should measure.
</td>
</tr><tr>
<td>
`key`
</td>
<td>
The string key of the measurement. If this is None, it defaults
to a comma-separated list of the target qubits' str values.
</td>
</tr><tr>
<td>
`invert_mask`
</td>
<td>
A list of Truthy or Falsey values indicating whether
the corresponding qubits should be flipped. None indicates no
inverting should be done.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An operation targeting the given qubits with a measurement.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError if the qubits are not instances of Qid.
</td>
</tr>

</table>

