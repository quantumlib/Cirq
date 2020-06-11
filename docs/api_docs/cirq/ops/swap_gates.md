<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.swap_gates" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.ops.swap_gates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/swap_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



SWAP and ISWAP gates.


This module creates Gate instances for the following gates:
    SWAP: the swap gate.
    ISWAP: a swap gate with a phase on the swapped subspace.

Each of these are implemented as EigenGates, which means that they can be
raised to a power (i.e. cirq.ISWAP**0.5). See the definition in EigenGate.

## Classes

[`class ISwapPowGate`](../../cirq/ops/ISwapPowGate.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`class SwapPowGate`](../../cirq/ops/SwapPowGate.md): The SWAP gate, possibly raised to a power. Exchanges qubits.

## Functions

[`ISWAP(...)`](../../cirq/ops/ISWAP.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`SWAP(...)`](../../cirq/ops/SWAP.md): The SWAP gate, possibly raised to a power. Exchanges qubits.

[`riswap(...)`](../../cirq/ops/riswap.md): Returns gate with matrix exp(+i angle_rads (X⊗X + Y⊗Y) / 2).

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
