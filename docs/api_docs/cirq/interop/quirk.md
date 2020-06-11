<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.interop.quirk" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.interop.quirk

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Code related to interoperating with Quirk, a drag-and-drop circuit simulator.



#### References:

https://github.com/strilanc/quirk - Quirk source code.
https://algassert.com/quirk - Live version of Quirk.


## Modules

[`cells`](../../cirq/interop/quirk/cells.md) module: This module defines building blocks for parsing Quirk circuits.

[`url_to_circuit`](../../cirq/interop/quirk/url_to_circuit.md) module

## Classes

[`class QuirkArithmeticOperation`](../../cirq/interop/quirk/QuirkArithmeticOperation.md): Applies arithmetic to a target and some inputs.

[`class QuirkInputRotationOperation`](../../cirq/interop/quirk/QuirkInputRotationOperation.md): Operates on target qubits in a way that varies based on an input qureg.

[`class QuirkQubitPermutationGate`](../../cirq/interop/quirk/QuirkQubitPermutationGate.md): A qubit permutation gate specified by a permutation list.

## Functions

[`quirk_json_to_circuit(...)`](../../cirq/interop/quirk_json_to_circuit.md): Constructs a Cirq circuit from Quirk's JSON format.

[`quirk_url_to_circuit(...)`](../../cirq/interop/quirk_url_to_circuit.md): Parses a Cirq circuit out of a Quirk URL.

