<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: cirq.google.optimizers

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Package for optimizers and gate compilers related to Google-specific devices.



## Modules

[`convert_to_sqrt_iswap`](../../cirq/google/optimizers/convert_to_sqrt_iswap.md) module

[`convert_to_sycamore_gates`](../../cirq/google/optimizers/convert_to_sycamore_gates.md) module

[`convert_to_xmon_gates`](../../cirq/google/optimizers/convert_to_xmon_gates.md) module

[`optimize_for_sycamore`](../../cirq/google/optimizers/optimize_for_sycamore.md) module: A combination of several optimizations targeting XmonDevice.

[`optimize_for_xmon`](../../cirq/google/optimizers/optimize_for_xmon.md) module: A combination of several optimizations targeting XmonDevice.

[`two_qubit_gates`](../../cirq/google/optimizers/two_qubit_gates.md) module

## Classes

[`class ConvertToSqrtIswapGates`](../../cirq/google/ConvertToSqrtIswapGates.md): Attempts to convert gates into ISWAP**-0.5 gates.

[`class ConvertToSycamoreGates`](../../cirq/google/ConvertToSycamoreGates.md): Attempts to convert non-native gates into SycamoreGates.

[`class ConvertToXmonGates`](../../cirq/google/ConvertToXmonGates.md): Attempts to convert strange gates into XmonGates.

[`class GateTabulation`](../../cirq/google/GateTabulation.md): A 2-qubit gate compiler based on precomputing/tabulating gate products.

## Functions

[`gate_product_tabulation(...)`](../../cirq/google/optimizers/gate_product_tabulation.md): Generate a GateTabulation for a base two qubit unitary.

[`optimized_for_sycamore(...)`](../../cirq/google/optimized_for_sycamore.md): Optimizes a circuit for Google devices.

[`optimized_for_xmon(...)`](../../cirq/google/optimized_for_xmon.md)

