<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.two_qubit_gates.gate_compilation.TwoQubitGateCompilation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# cirq.google.optimizers.two_qubit_gates.gate_compilation.TwoQubitGateCompilation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/two_qubit_gates/gate_compilation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Represents a compilation of a target 2-qubit with respect to a base gate.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimizers.two_qubit_gates.gate_compilation.TwoQubitGateCompilation(
    _cls, base_gate_unitary, target_gate, local_unitaries, actual_gate, success
)
</code></pre>



<!-- Placeholder for "Used in" -->

This object encodes the relationship between 4x4 unitary operators

U_target ~ k_N · U_base · k_{N-1} · ... · k_1 · U_base · k_0

where U_target, U_base are 2-local and k_j are 1-local.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`base_gate_unitary`
</td>
<td>

</td>
</tr><tr>
<td>
`target_gate`
</td>
<td>
4x4 unitary denoting U_target above.
</td>
</tr><tr>
<td>
`local_unitaries`
</td>
<td>
Sequence of 2-tuples (k_{00},k_{01}),(k_{10},k_{11})...
where k_j = k_{j0} ⊗ k_{j1} in the product above. Each k_{j0},
k_{j1} is a 2x2 unitary.
</td>
</tr><tr>
<td>
`actual_gate`
</td>
<td>
4x4 unitary denoting the right hand side above, ideally
equal to U_target.
</td>
</tr><tr>
<td>
`success`
</td>
<td>
Whether actual_gate is expected to be close to U_target.
</td>
</tr><tr>
<td>
`base_gate`
</td>
<td>
4x4 unitary denoting U_base above.
</td>
</tr>
</table>



