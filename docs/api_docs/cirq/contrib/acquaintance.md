<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.acquaintance" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="UnconstrainedAcquaintanceDevice"/>
</div>

# Module: cirq.contrib.acquaintance

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/acquaintance/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tools for creating and using acquaintance strategies.



## Modules

[`bipartite`](../../cirq/contrib/acquaintance/bipartite.md) module

[`devices`](../../cirq/contrib/acquaintance/devices.md) module

[`executor`](../../cirq/contrib/acquaintance/executor.md) module

[`gates`](../../cirq/contrib/acquaintance/gates.md) module

[`inspection_utils`](../../cirq/contrib/acquaintance/inspection_utils.md) module

[`mutation_utils`](../../cirq/contrib/acquaintance/mutation_utils.md) module

[`optimizers`](../../cirq/contrib/acquaintance/optimizers.md) module

[`permutation`](../../cirq/contrib/acquaintance/permutation.md) module

[`shift`](../../cirq/contrib/acquaintance/shift.md) module

[`shift_swap_network`](../../cirq/contrib/acquaintance/shift_swap_network.md) module

[`strategies`](../../cirq/contrib/acquaintance/strategies.md) module: Acquaintance strategies.

[`testing`](../../cirq/contrib/acquaintance/testing.md) module

[`topological_sort`](../../cirq/contrib/acquaintance/topological_sort.md) module

## Classes

[`class AcquaintanceOperation`](../../cirq/contrib/acquaintance/AcquaintanceOperation.md): Represents an a acquaintance opportunity between a particular set of

[`class AcquaintanceOpportunityGate`](../../cirq/contrib/acquaintance/AcquaintanceOpportunityGate.md): Represents an acquaintance opportunity. An acquaintance opportunity is

[`class BipartiteGraphType`](../../cirq/contrib/acquaintance/BipartiteGraphType.md): An enumeration.

[`class BipartiteSwapNetworkGate`](../../cirq/contrib/acquaintance/BipartiteSwapNetworkGate.md): A swap network that acquaints qubits in one half with qubits in the

[`class CircularShiftGate`](../../cirq/contrib/acquaintance/CircularShiftGate.md): Performs a cyclical permutation of the qubits to the left by a specified

[`class GreedyExecutionStrategy`](../../cirq/contrib/acquaintance/GreedyExecutionStrategy.md): A greedy execution strategy.

[`class LinearPermutationGate`](../../cirq/contrib/acquaintance/LinearPermutationGate.md): A permutation gate that decomposes a given permutation using a linear

[`class PermutationGate`](../../cirq/contrib/acquaintance/PermutationGate.md): A permutation gate indicates a change in the mapping from qubits to

[`class ShiftSwapNetworkGate`](../../cirq/contrib/acquaintance/ShiftSwapNetworkGate.md): A swap network that generalizes the circular shift gate.

[`class StrategyExecutor`](../../cirq/contrib/acquaintance/StrategyExecutor.md): Executes an acquaintance strategy.

[`class SwapNetworkGate`](../../cirq/contrib/acquaintance/SwapNetworkGate.md): A single gate representing a generalized swap network.

[`class SwapPermutationGate`](../../cirq/contrib/acquaintance/SwapPermutationGate.md): Generic swap gate.

## Functions

[`DECOMPOSE_PERMUTATION_GATES(...)`](../../cirq/contrib/acquaintance/DECOMPOSE_PERMUTATION_GATES.md)

[`EXPAND_PERMUTATION_GATES(...)`](../../cirq/contrib/acquaintance/EXPAND_PERMUTATION_GATES.md)

[`acquaint(...)`](../../cirq/contrib/acquaintance/acquaint.md)

[`complete_acquaintance_strategy(...)`](../../cirq/contrib/acquaintance/complete_acquaintance_strategy.md): Returns an acquaintance strategy capable of executing a gate corresponding

[`cubic_acquaintance_strategy(...)`](../../cirq/contrib/acquaintance/cubic_acquaintance_strategy.md): Acquaints every triple of qubits.

[`display_mapping(...)`](../../cirq/contrib/acquaintance/display_mapping.md): Inserts display gates between moments to indicate the mapping throughout

[`expose_acquaintance_gates(...)`](../../cirq/contrib/acquaintance/expose_acquaintance_gates.md): Decomposes any permutation gates that provide acquaintance opportunities

[`get_acquaintance_size(...)`](../../cirq/contrib/acquaintance/get_acquaintance_size.md): The maximum number of qubits to be acquainted with each other.

[`get_logical_acquaintance_opportunities(...)`](../../cirq/contrib/acquaintance/get_logical_acquaintance_opportunities.md)

[`get_logical_operations(...)`](../../cirq/contrib/acquaintance/get_logical_operations.md): Gets the logical operations specified by the physical operations and

[`is_topologically_sorted(...)`](../../cirq/contrib/acquaintance/is_topologically_sorted.md): Whether a given order of operations is consistent with the DAG.

[`quartic_paired_acquaintance_strategy(...)`](../../cirq/contrib/acquaintance/quartic_paired_acquaintance_strategy.md): Acquaintance strategy for pairs of pairs.

[`random_topological_sort(...)`](../../cirq/contrib/acquaintance/random_topological_sort.md)

[`rectify_acquaintance_strategy(...)`](../../cirq/contrib/acquaintance/rectify_acquaintance_strategy.md): Splits moments so that they contain either only acquaintance gates

[`remove_redundant_acquaintance_opportunities(...)`](../../cirq/contrib/acquaintance/remove_redundant_acquaintance_opportunities.md): Removes redundant acquaintance opportunities.

[`replace_acquaintance_with_swap_network(...)`](../../cirq/contrib/acquaintance/replace_acquaintance_with_swap_network.md): Replace every moment containing acquaintance gates (after

[`return_to_initial_mapping(...)`](../../cirq/contrib/acquaintance/return_to_initial_mapping.md)

[`update_mapping(...)`](../../cirq/contrib/acquaintance/update_mapping.md): Updates a mapping (in place) from qubits to logical indices according to

[`uses_consistent_swap_gate(...)`](../../cirq/contrib/acquaintance/uses_consistent_swap_gate.md)

## Other Members

* `UnconstrainedAcquaintanceDevice` <a id="UnconstrainedAcquaintanceDevice"></a>
