<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance.permutation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="LogicalIndex"/>
<meta itemprop="property" content="LogicalMappingKey"/>
<meta itemprop="property" content="TYPE_CHECKING"/>
</div>

# Module: cirq.contrib.acquaintance.permutation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/permutation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>







## Classes

[`class DecomposePermutationGates`](../../../cirq/contrib/acquaintance/permutation/DecomposePermutationGates.md): An optimizer that expands composite operations via <a href="../../../cirq/protocols/decompose.md"><code>cirq.decompose</code></a>.

[`class LinearPermutationGate`](../../../cirq/contrib/acquaintance/LinearPermutationGate.md): A permutation gate that decomposes a given permutation using a linear

[`class MappingDisplayGate`](../../../cirq/contrib/acquaintance/permutation/MappingDisplayGate.md): Displays the indices mapped to a set of wires.

[`class PermutationGate`](../../../cirq/contrib/acquaintance/PermutationGate.md): A permutation gate indicates a change in the mapping from qubits to

[`class SwapPermutationGate`](../../../cirq/contrib/acquaintance/SwapPermutationGate.md): Generic swap gate.

## Functions

[`DECOMPOSE_PERMUTATION_GATES(...)`](../../../cirq/contrib/acquaintance/DECOMPOSE_PERMUTATION_GATES.md)

[`EXPAND_PERMUTATION_GATES(...)`](../../../cirq/contrib/acquaintance/EXPAND_PERMUTATION_GATES.md)

[`display_mapping(...)`](../../../cirq/contrib/acquaintance/display_mapping.md): Inserts display gates between moments to indicate the mapping throughout

[`get_logical_operations(...)`](../../../cirq/contrib/acquaintance/get_logical_operations.md): Gets the logical operations specified by the physical operations and

[`return_to_initial_mapping(...)`](../../../cirq/contrib/acquaintance/return_to_initial_mapping.md)

[`update_mapping(...)`](../../../cirq/contrib/acquaintance/update_mapping.md): Updates a mapping (in place) from qubits to logical indices according to

[`uses_consistent_swap_gate(...)`](../../../cirq/contrib/acquaintance/uses_consistent_swap_gate.md)

## Type Aliases

[`LogicalGates`](../../../cirq/contrib/acquaintance/executor/LogicalGates.md)

[`LogicalIndexSequence`](../../../cirq/contrib/acquaintance/executor/LogicalIndexSequence.md)

[`LogicalMapping`](../../../cirq/contrib/acquaintance/executor/LogicalMapping.md)

## Other Members

* `LogicalIndex` <a id="LogicalIndex"></a>
* `LogicalMappingKey` <a id="LogicalMappingKey"></a>
* `TYPE_CHECKING = False` <a id="TYPE_CHECKING"></a>
