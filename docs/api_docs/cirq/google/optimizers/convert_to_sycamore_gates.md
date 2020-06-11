<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimizers.convert_to_sycamore_gates" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="PAULI_OPS"/>
<meta itemprop="property" content="TYPE_CHECKING"/>
<meta itemprop="property" content="UNITARY_ZZ"/>
</div>

# Module: cirq.google.optimizers.convert_to_sycamore_gates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/convert_to_sycamore_gates.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Classes

[`class ConvertToSycamoreGates`](../../../cirq/google/ConvertToSycamoreGates.md): Attempts to convert non-native gates into SycamoreGates.

## Functions

[`cphase(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/cphase.md): Implement a cphase using the Ising gate generated from 2 Sycamore gates

[`create_corrected_circuit(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/create_corrected_circuit.md)

[`decompose_arbitrary_into_syc_analytic(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/decompose_arbitrary_into_syc_analytic.md): Synthesize an arbitrary 2 qubit operation to a sycamore operation using

[`decompose_arbitrary_into_syc_tabulation(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/decompose_arbitrary_into_syc_tabulation.md): Synthesize an arbitrary 2 qubit operation to a sycamore operation using

[`decompose_cz_into_syc(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/decompose_cz_into_syc.md): Decompose CZ into sycamore gates using precomputed coefficients

[`decompose_iswap_into_syc(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/decompose_iswap_into_syc.md): Decompose ISWAP into sycamore gates using precomputed coefficients

[`decompose_phased_iswap_into_syc(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/decompose_phased_iswap_into_syc.md): Decompose PhasedISwap with an exponent of 1.

[`decompose_phased_iswap_into_syc_precomputed(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/decompose_phased_iswap_into_syc_precomputed.md): Decompose PhasedISwap into sycamore gates using precomputed coefficients.

[`decompose_swap_into_syc(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/decompose_swap_into_syc.md): Decompose SWAP into sycamore gates using precomputed coefficients

[`find_local_equivalents(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/find_local_equivalents.md): Given two unitaries with the same interaction coefficients but different

[`known_two_q_operations_to_sycamore_operations(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/known_two_q_operations_to_sycamore_operations.md): Synthesize a known gate operation to a sycamore operation

[`rzz(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/rzz.md): Generate exp(-1j * theta * zz) from Sycamore gates.

[`swap_rzz(...)`](../../../cirq/google/optimizers/convert_to_sycamore_gates/swap_rzz.md): An implementation of SWAP * EXP(1j theta ZZ) using three sycamore gates.

## Other Members

* `PAULI_OPS` <a id="PAULI_OPS"></a>
* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
* `UNITARY_ZZ` <a id="UNITARY_ZZ"></a>
