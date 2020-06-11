<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.fsim_gate" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.ops.fsim_gate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/fsim_gate.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Defines the fermionic simulation gate family.


This is the family of two-qubit gates that preserve excitations (number of ON
qubits), ignoring single-qubit gates and global phase. For example, when using
the second quantized representation of electrons to simulate chemistry, this is
a natural gateset because each ON qubit corresponds to an electron and in the
context of chemistry the electron count is conserved over time. This property
applies more generally to fermions, thus the name of the gate.

## Classes

[`class FSimGate`](../../cirq/ops/FSimGate.md): Fermionic simulation gate family.

