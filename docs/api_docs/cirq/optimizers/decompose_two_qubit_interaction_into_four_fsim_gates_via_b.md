<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.decompose_two_qubit_interaction_into_four_fsim_gates_via_b" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.decompose_two_qubit_interaction_into_four_fsim_gates_via_b

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/two_qubit_to_fsim.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decomposes operations into an FSimGate near theta=pi/2, phi=0.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.decompose_two_qubit_interaction_into_four_fsim_gates_via_b`, `cirq.optimizers.two_qubit_to_fsim.decompose_two_qubit_interaction_into_four_fsim_gates_via_b`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.decompose_two_qubit_interaction_into_four_fsim_gates_via_b(
    interaction: Union['cirq.Operation', 'cirq.Gate', np.ndarray, Any],
    *,
    fsim_gate: Union['cirq.FSimGate', 'cirq.ISwapPowGate'] = None,
    qubits: Sequence['cirq.Qid'] = None
) -> "cirq.Circuit"
</code></pre>



<!-- Placeholder for "Used in" -->

This decomposition is guaranteed to use exactly four of the given FSim
gates. It works by decomposing into two B gates and then decomposing each
B gate into two of the given FSim gate.

This decomposition only works for FSim gates with a theta (iswap angle)
between 3/8π and 5/8π (i.e. within 22.5° of maximum strength) and a
phi (cphase angle) between -π/4 and +π/4 (i.e. within 45° of minimum
strength).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`interaction`
</td>
<td>
The two qubit operation to synthesize. This can either be
a cirq object (such as a gate, operation, or circuit) or a raw numpy
array specifying the 4x4 unitary matrix.
</td>
</tr><tr>
<td>
`fsim_gate`
</td>
<td>
The only two qubit gate that is permitted to appear in the
output. Must satisfy 3/8π < phi < 5/8π and abs(theta) < pi/4.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
The qubits that the resulting operations should apply the
desired interaction to. If not set then defaults to either the
qubits of the given interaction (if it is a <a href="../../cirq/ops/Operation.md"><code>cirq.Operation</code></a>) or
else to <a href="../../cirq/devices/LineQubit.md#range"><code>cirq.LineQubit.range(2)</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of operations implementing the desired two qubit unitary. The
list will include four operations of the given fsim gate, various single
qubit operations, and a global phase operation.
</td>
</tr>

</table>

