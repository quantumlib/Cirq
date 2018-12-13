# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for creating and using acquaintance strategies."""

from cirq.contrib.acquaintance.bipartite import (
        BipartiteGraphType, BipartiteSwapNetworkGate)

from cirq.contrib.acquaintance.devices import (
        get_acquaintance_size, UnconstrainedAcquaintanceDevice)

from cirq.contrib.acquaintance.executor import (
        AcquaintanceOperation, GreedyExecutionStrategy, StrategyExecutor)

from cirq.contrib.acquaintance.gates import (
        ACQUAINT, AcquaintanceOpportunityGate, SwapNetworkGate)

from cirq.contrib.acquaintance.inspection_utils import (
        get_logical_acquaintance_opportunities)

from cirq.contrib.acquaintance.mutation_utils import (
    rectify_acquaintance_strategy,
    replace_acquaintance_with_swap_network)

from cirq.contrib.acquaintance.permutation import (
        LinearPermutationGate, PermutationGate,
        SwapPermutationGate, update_mapping)

from cirq.contrib.acquaintance.shift import (
        CircularShiftGate)

from cirq.contrib.acquaintance.strategies import (
    complete_acquaintance_strategy)

