<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.gate_product_tabulation" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimizers.gate_product_tabulation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/gate_compilation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generate a GateTabulation for a base two qubit unitary.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.optimizers.two_qubit_gates.gate_compilation.gate_product_tabulation`, `cirq.google.optimizers.two_qubit_gates.gate_product_tabulation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.gate_product_tabulation(
    base_gate: np.ndarray,
    max_infidelity: float,
    *,
    sample_scaling: int = 50,
    allow_missed_points: bool = True,
    random_state: "cirq.RANDOM_STATE_OR_SEED_LIKE" = None
) -> <a href="../../../cirq/google/GateTabulation.md"><code>cirq.google.GateTabulation</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`base_gate`
</td>
<td>
The base gate of the tabulation.
</td>
</tr><tr>
<td>
`max_infidelity`
</td>
<td>
Sets the desired density of tabulated product unitaries.
The typical nearest neighbor Euclidean spacing (of the KAK vectors)
will be on the order of \sqrt(max_infidelity). Thus the number of
tabulated points will scale as max_infidelity^{-3/2}.
</td>
</tr><tr>
<td>
`sample_scaling`
</td>
<td>
Relative number of random gate products to use in the
tabulation. The total number of random local unitaries scales as
~ max_infidelity^{-3/2} * sample_scaling. Must be positive.
</td>
</tr><tr>
<td>
`random_state`
</td>
<td>
Random state or random state seed.
</td>
</tr><tr>
<td>
`allow_missed_points`
</td>
<td>
If True, the tabulation is allowed to conclude
even if not all points in the Weyl chamber are expected to be
compilable using 2 or 3 base gates. Otherwise an error is raised
in this case.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A GateTabulation object used to compile new two-qubit gates from
products of the base gate with 1-local unitaries.
</td>
</tr>

</table>

