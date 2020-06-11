<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.gate_features" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.ops.gate_features

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/gate_features.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Marker classes for indicating which additional features gates support.


For example: some gates are reversible, some have known matrices, etc.

## Classes

[`class InterchangeableQubitsGate`](../../cirq/ops/InterchangeableQubitsGate.md): Indicates operations should be equal under some qubit permutations.

[`class SingleQubitGate`](../../cirq/ops/SingleQubitGate.md): A gate that must be applied to exactly one qubit.

[`class ThreeQubitGate`](../../cirq/ops/ThreeQubitGate.md): A gate that must be applied to exactly three qubits.

[`class TwoQubitGate`](../../cirq/ops/TwoQubitGate.md): A gate that must be applied to exactly two qubits.

