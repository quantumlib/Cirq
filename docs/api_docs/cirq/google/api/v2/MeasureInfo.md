<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.api.v2.MeasureInfo" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# cirq.google.api.v2.MeasureInfo

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/api/v2/results.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Extra info about a single measurement within a circuit.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.api.v2.results.MeasureInfo`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.api.v2.MeasureInfo(
    _cls, key, qubits, slot, invert_mask
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`key`
</td>
<td>
String identifying this measurement.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
List of measured qubits, in order.
</td>
</tr><tr>
<td>
`slot`
</td>
<td>
The location of this measurement within the program. For circuits,
this is just the moment index; for schedules it is the start time
of the measurement. This is used internally when scheduling on
hardware so that we can combine measurements that occupy the same
slot.
</td>
</tr><tr>
<td>
`invert_mask`
</td>
<td>
a list of booleans describing whether the results should
be flipped for each of the qubits in the qubits field.
</td>
</tr>
</table>



