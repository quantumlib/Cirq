<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimized_for_sycamore" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimized_for_sycamore

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/optimize_for_sycamore.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Optimizes a circuit for Google devices.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.optimizers.optimize_for_sycamore.optimized_for_sycamore`, `cirq.google.optimizers.optimized_for_sycamore`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimized_for_sycamore(
    circuit: "cirq.Circuit",
    *,
    new_device: Optional['cirq.google.XmonDevice'] = None,
    qubit_map: Callable[['cirq.Qid'], <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>] = (lambda e: cast(devices.GridQubit, e)),
    optimizer_type: str = 'sqrt_iswap',
    tolerance: float = 1e-05,
    tabulation_resolution: Optional[float] = None
) -> "cirq.Circuit"
</code></pre>



<!-- Placeholder for "Used in" -->

Uses a set of optimizers that will compile to the proper gateset for the
device (xmon, sqrt_iswap, or sycamore gates) and then use optimizers to
compresss the gate depth down as much as is easily algorithmically possible
by merging rotations, ejecting Z gates, etc.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`circuit`
</td>
<td>
The circuit to optimize.
</td>
</tr><tr>
<td>
`new_device`
</td>
<td>
The device the optimized circuit should be targeted at. If
set to None, the circuit's current device is used.
</td>
</tr><tr>
<td>
`qubit_map`
</td>
<td>
Transforms the qubits (e.g. so that they are GridQubits).
</td>
</tr><tr>
<td>
`optimizer_type`
</td>
<td>
A string defining the optimizations to apply.
Possible values are  'xmon', 'xmon_partial_cz', 'sqrt_iswap',
'sycamore'
</td>
</tr><tr>
<td>
`tolerance`
</td>
<td>
The tolerance passed to the various circuit optimization
passes.
</td>
</tr><tr>
<td>
`tabulation_resolution`
</td>
<td>
If provided, compute a gateset tabulation
with the specified resolution and use it to approximately
compile arbitrary two-qubit gates for which an analytic compilation
is not known.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The optimized circuit.
</td>
</tr>

</table>

