# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for creating and using acquaintance strategies."""
from cirq.contrib.acquaintance.bipartite import (
    BipartiteGraphType as BipartiteGraphType,
    BipartiteSwapNetworkGate as BipartiteSwapNetworkGate,
)

from cirq.contrib.acquaintance.devices import (
    get_acquaintance_size as get_acquaintance_size,
    UnconstrainedAcquaintanceDevice as UnconstrainedAcquaintanceDevice,
)

from cirq.contrib.acquaintance.executor import (
    AcquaintanceOperation as AcquaintanceOperation,
    GreedyExecutionStrategy as GreedyExecutionStrategy,
    StrategyExecutorTransformer as StrategyExecutorTransformer,
)

from cirq.contrib.acquaintance.gates import (
    acquaint as acquaint,
    AcquaintanceOpportunityGate as AcquaintanceOpportunityGate,
    SwapNetworkGate as SwapNetworkGate,
)

from cirq.contrib.acquaintance.inspection_utils import (
    get_logical_acquaintance_opportunities as get_logical_acquaintance_opportunities,
)

from cirq.contrib.acquaintance.mutation_utils import (
    expose_acquaintance_gates as expose_acquaintance_gates,
    rectify_acquaintance_strategy as rectify_acquaintance_strategy,
    replace_acquaintance_with_swap_network as replace_acquaintance_with_swap_network,
)

from cirq.contrib.acquaintance.optimizers import (
    remove_redundant_acquaintance_opportunities as remove_redundant_acquaintance_opportunities,
)

from cirq.contrib.acquaintance.permutation import (
    LinearPermutationGate as LinearPermutationGate,
    PermutationGate as PermutationGate,
    SwapPermutationGate as SwapPermutationGate,
    update_mapping as update_mapping,
    get_logical_operations as get_logical_operations,
    display_mapping as display_mapping,
    return_to_initial_mapping as return_to_initial_mapping,
    uses_consistent_swap_gate as uses_consistent_swap_gate,
    EXPAND_PERMUTATION_GATES as EXPAND_PERMUTATION_GATES,
    DECOMPOSE_PERMUTATION_GATES as DECOMPOSE_PERMUTATION_GATES,
)

from cirq.contrib.acquaintance.shift import CircularShiftGate as CircularShiftGate

from cirq.contrib.acquaintance.shift_swap_network import (
    ShiftSwapNetworkGate as ShiftSwapNetworkGate,
)

from cirq.contrib.acquaintance.strategies import (
    complete_acquaintance_strategy as complete_acquaintance_strategy,
    cubic_acquaintance_strategy as cubic_acquaintance_strategy,
    quartic_paired_acquaintance_strategy as quartic_paired_acquaintance_strategy,
)

from cirq.contrib.acquaintance.topological_sort import (
    is_topologically_sorted as is_topologically_sorted,
    random_topological_sort as random_topological_sort,
)

from cirq.contrib.acquaintance import testing
