<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.generate_boixo_2018_supremacy_circuits_v2_bristlecone" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.experiments.generate_boixo_2018_supremacy_circuits_v2_bristlecone

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/google_v2_supremacy_circuit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generates Google Random Circuits v2 in Bristlecone.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.google_v2_supremacy_circuit.generate_boixo_2018_supremacy_circuits_v2_bristlecone`, `cirq.generate_boixo_2018_supremacy_circuits_v2_bristlecone`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.generate_boixo_2018_supremacy_circuits_v2_bristlecone(
    n_rows: int,
    cz_depth: int,
    seed: int
) -> <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->
See also https://arxiv.org/abs/1807.10749

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`n_rows`
</td>
<td>
number of rows in a Bristlecone lattice.
Note that we do not include single qubit corners.
</td>
</tr><tr>
<td>
`cz_depth`
</td>
<td>
number of layers with CZ gates.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
seed for the random instance.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A circuit with given size and seed.
</td>
</tr>

</table>

