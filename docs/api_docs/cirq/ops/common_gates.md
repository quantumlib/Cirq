<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.common_gates" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.ops.common_gates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/common_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Quantum gates that are commonly used in the literature.


This module creates Gate instances for the following gates:
    X,Y,Z: Pauli gates.
    H,S: Clifford gates.
    T: A non-Clifford gate.
    CZ: Controlled phase gate.
    CNOT: Controlled not gate.

Each of these are implemented as EigenGates, which means that they can be
raised to a power (i.e. cirq.H**0.5). See the definition in EigenGate.

## Classes

[`class CNotPowGate`](../../cirq/ops/CNotPowGate.md): A gate that applies a controlled power of an X gate.

[`class CXPowGate`](../../cirq/ops/CNotPowGate.md): A gate that applies a controlled power of an X gate.

[`class CZPowGate`](../../cirq/ops/CZPowGate.md): A gate that applies a phase to the |11⟩ state of two qubits.

[`class HPowGate`](../../cirq/ops/HPowGate.md): A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

[`class XPowGate`](../../cirq/ops/XPowGate.md): A gate that rotates around the X axis of the Bloch sphere.

[`class YPowGate`](../../cirq/ops/YPowGate.md): A gate that rotates around the Y axis of the Bloch sphere.

[`class ZPowGate`](../../cirq/ops/ZPowGate.md): A gate that rotates around the Z axis of the Bloch sphere.

## Functions

[`CNOT(...)`](../../cirq/ops/CNOT.md): A gate that applies a controlled power of an X gate.

[`CX(...)`](../../cirq/ops/CNOT.md): A gate that applies a controlled power of an X gate.

[`CZ(...)`](../../cirq/ops/CZ.md): A gate that applies a phase to the |11⟩ state of two qubits.

[`H(...)`](../../cirq/ops/H.md): A Gate that performs a rotation around the X+Z axis of the Bloch sphere.

[`ISWAP(...)`](../../cirq/ops/ISWAP.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`S(...)`](../../cirq/ops/S.md): A gate that rotates around the Z axis of the Bloch sphere.

[`SWAP(...)`](../../cirq/ops/SWAP.md): The SWAP gate, possibly raised to a power. Exchanges qubits.

[`T(...)`](../../cirq/ops/T.md): A gate that rotates around the Z axis of the Bloch sphere.

[`rx(...)`](../../cirq/ops/rx.md): Returns a gate with the matrix e^{-i X rads / 2}.

[`ry(...)`](../../cirq/ops/ry.md): Returns a gate with the matrix e^{-i Y rads / 2}.

[`rz(...)`](../../cirq/ops/rz.md): Returns a gate with the matrix e^{-i Z rads / 2}.

