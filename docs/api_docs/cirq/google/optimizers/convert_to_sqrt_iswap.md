<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sqrt_iswap" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.google.optimizers.convert_to_sqrt_iswap

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sqrt_iswap.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Classes

[`class ConvertToSqrtIswapGates`](../../../cirq/google/ConvertToSqrtIswapGates.md): Attempts to convert gates into ISWAP**-0.5 gates.

## Functions

[`SQRT_ISWAP(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/SQRT_ISWAP.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`SQRT_ISWAP_INV(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/SQRT_ISWAP_INV.md): Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.

[`cphase_symbols_to_sqrt_iswap(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/cphase_symbols_to_sqrt_iswap.md): Version of cphase_to_sqrt_iswap that works with symbols.

[`cphase_to_sqrt_iswap(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/cphase_to_sqrt_iswap.md): Implement a C-Phase gate using two sqrt ISWAP gates and single-qubit

[`fsim_gate(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/fsim_gate.md): FSimGate has a default decomposition in cirq to XXPowGate and YYPowGate,

[`is_basic_gate(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/is_basic_gate.md): Check if a gate is a basic supported one-qubit gate.

[`is_sqrt_iswap(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/is_sqrt_iswap.md): Checks if this is a ± sqrt(iSWAP) gate specified using either

[`is_sqrt_iswap_compatible(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/is_sqrt_iswap_compatible.md): Check if the given operation is compatible with the sqrt_iswap gateset

[`iswap_to_sqrt_iswap(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/iswap_to_sqrt_iswap.md): Implement the evolution of the hopping term using two sqrt_iswap gates

[`swap_to_sqrt_iswap(...)`](../../../cirq/google/optimizers/convert_to_sqrt_iswap/swap_to_sqrt_iswap.md): Implement the evolution of the hopping term using two sqrt_iswap gates

## Other Members

* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
