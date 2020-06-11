<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/circuit_compare.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Determines if two circuits have equivalent effects.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.circuit_compare.assert_circuits_with_terminal_measurements_are_equivalent`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
    actual: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    reference: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    atol: float
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

The circuits can contain measurements, but the measurements must be at the
end of the circuit. Circuits are equivalent if, for all possible inputs,
their outputs (classical bits for lines terminated with measurement and
qubits for lines without measurement) are observationally indistinguishable
up to a tolerance. Note that under this definition of equivalence circuits
that differ solely in the overall phase of the post-measurement state of
measured qubits are considered equivalent.

For example, applying an extra Z gate to an unmeasured qubit changes the
effect of a circuit. But inserting a Z gate operation just before a
measurement does not.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`actual`
</td>
<td>
The circuit that was actually computed by some process.
</td>
</tr><tr>
<td>
`reference`
</td>
<td>
A circuit with the correct function.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Absolute error tolerance.
</td>
</tr>
</table>

