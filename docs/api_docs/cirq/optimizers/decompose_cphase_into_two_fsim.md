<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.optimizers.decompose_cphase_into_two_fsim" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.optimizers.decompose_cphase_into_two_fsim

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/optimizers/cphase_to_fsim.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decomposes CZPowGate into two FSimGates.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.decompose_cphase_into_two_fsim`, `cirq.optimizers.cphase_to_fsim.decompose_cphase_into_two_fsim`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.optimizers.decompose_cphase_into_two_fsim(
    cphase_gate: "cirq.CZPowGate",
    *,
    fsim_gate: "cirq.FSimGate" = None,
    qubits: Optional[Sequence['cirq.Qid']] = 1e-08,
    atol: float = 1e-08
) -> "cirq.OP_TREE"
</code></pre>



<!-- Placeholder for "Used in" -->

This function implements the decomposition described in section VII F I
of https://arxiv.org/abs/1910.11333.

The decomposition results in exactly two FSimGates and a few single-qubit
rotations. It is feasible if and only if one of the following conditions
is met:

    |sin(θ)| <= |sin(δ/4)| <= |sin(φ/2)|
    |sin(φ/2)| <= |sin(δ/4)| <= |sin(θ)|

where:

     θ = fsim_gate.theta,
     φ = fsim_gate.phi,
     δ = -π * cphase_gate.exponent.

Note that the gate parametrizations are non-injective. For the
decomposition to be feasible it is sufficient that one of the
parameter values which correspond to the provided gate satisfies
the constraints. This function will find and use the appropriate
value whenever it exists.

The constraints above imply that certain FSimGates are not suitable
for use in this decomposition regardless of the target CZPowGate. We
reject such gates based on how close |sin(θ)| is to |sin(φ/2)|, see
atol argument below.

This implementation accounts for the global phase.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cphase_gate`
</td>
<td>
The CZPowGate to synthesize.
</td>
</tr><tr>
<td>
`fsim_gate`
</td>
<td>
The only two qubit gate that is permitted to appear in the
output.
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
The qubits to apply the resulting operations to. If not set,
defaults <a href="../../cirq/devices/LineQubit.md#range"><code>cirq.LineQubit.range(2)</code></a>.
</td>
</tr><tr>
<td>
`atol`
</td>
<td>
Tolerance used to determine whether fsim_gate is valid. The gate
is invalid if the squares of the sines of the theta angle and half
the phi angle are too close.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Operations equivalent to cphase_gate and consisting solely of two copies
of fsim_gate and a few single-qubit rotations.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError under any of the following circumstances:
* cphase_gate or fsim_gate is parametrized,
* cphase_gate and fsim_gate do not satisfy the conditions above,
* fsim_gate has invalid angles (see atol argument above),
* incorrect number of qubits are provided.
</td>
</tr>

</table>

