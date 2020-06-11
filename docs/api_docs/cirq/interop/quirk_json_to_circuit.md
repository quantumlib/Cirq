<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.interop.quirk_json_to_circuit" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.interop.quirk_json_to_circuit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/url_to_circuit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Constructs a Cirq circuit from Quirk's JSON format.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.interop.quirk.quirk_json_to_circuit`, `cirq.interop.quirk.url_to_circuit.quirk_json_to_circuit`, `cirq.quirk_json_to_circuit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.interop.quirk_json_to_circuit(
    data: <a href="../../cirq/circuits/CircuitDag/adjlist_inner_dict_factory.md"><code>cirq.circuits.CircuitDag.adjlist_inner_dict_factory</code></a>,
    *,
    qubits: Optional[Sequence['cirq.Qid']] = None,
    extra_cell_makers: Union[Dict[str, 'cirq.Gate'], Iterable['cirq.interop.quirk.cells.CellMaker']] = (),
    quirk_url: Optional[str] = None,
    max_operation_count: int = (10 ** 6)
) -> "cirq.Circuit"
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`
</td>
<td>
Data parsed from quirk's JSON representation.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
Qubits to use in the circuit. See quirk_url_to_circuit.
</td>
</tr><tr>
<td>
`extra_cell_makers`
</td>
<td>
Non-standard Quirk cells to accept. See
quirk_url_to_circuit.
</td>
</tr><tr>
<td>
`quirk_url`
</td>
<td>
If given, the original URL from which the JSON was parsed, as
described in quirk_url_to_circuit.
</td>
</tr><tr>
<td>
`max_operation_count`
</td>
<td>
If the number of operations in the circuit would
exceed this value, the method raises a `ValueError` instead of
attempting to construct the circuit. This is important to specify
for servers parsing unknown input, because Quirk's format allows for
a billion laughs attack in the form of nested custom gates.
</td>
</tr>
</table>



#### Examples:

>>> print(cirq.quirk_json_to_circuit(
...     {"cols":[["H"], ["•", "X"]]}
... ))

* <b>`0`</b>: ───H───@───
          │
* <b>`1`</b>: ───────X───


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The parsed circuit.
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
Invalid circuit URL, or circuit would be larger than
`max_operations_count`.
</td>
</tr>
</table>

