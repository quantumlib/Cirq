<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.GateTabulation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="compile_two_qubit_gate"/>
</div>

# cirq.google.GateTabulation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/gate_compilation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A 2-qubit gate compiler based on precomputing/tabulating gate products.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.optimizers.GateTabulation`, `cirq.google.optimizers.two_qubit_gates.GateTabulation`, `cirq.google.optimizers.two_qubit_gates.gate_compilation.GateTabulation`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.GateTabulation(
    base_gate, kak_vecs, single_qubit_gates, max_expected_infidelity, summary,
    missed_points
)
</code></pre>



<!-- Placeholder for "Used in" -->

    

## Methods

<h3 id="compile_two_qubit_gate"><code>compile_two_qubit_gate</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/gate_compilation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compile_two_qubit_gate(
    unitary: np.ndarray
) -> <a href="../../cirq/google/optimizers/two_qubit_gates/gate_compilation/TwoQubitGateCompilation.md"><code>cirq.google.optimizers.two_qubit_gates.gate_compilation.TwoQubitGateCompilation</code></a>
</code></pre>

Compute single qubit gates required to compile a desired unitary.

Given a desired unitary U, this computes the sequence of 1-local gates
k_j such that the product

k_{n-1} A k_{n-2} A ... k_1 A k_0

is close to U. Here A is the base_gate of the tabulation.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`unitary`
</td>
<td>
Unitary (U above) to compile.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A TwoQubitGateCompilation object encoding the required local
unitaries and resulting product above.
</td>
</tr>

</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/gate_compilation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




