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

from typing import cast, Sequence, TYPE_CHECKING

from cirq import circuits, ops

from cirq.contrib.acquaintance.devices import (
    is_acquaintance_strategy)
from cirq.contrib.acquaintance.gates import ACQUAINT
from cirq.contrib.acquaintance.executor import (
    AcquaintanceOperation)
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
from cirq.contrib.acquaintance.inspection_utils import LogicalAnnotator

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import FrozenSet, List, Set


def remove_redundant_acquaintance_opportunities(
        strategy: circuits.Circuit) -> None:
    if not is_acquaintance_strategy(strategy):
        raise TypeError('not is_acquaintance_strategy(circuit)')

    qubits = tuple(strategy.all_qubits())
    mapping = {q: i for i, q in enumerate(qubits)}

    expose_acquaintance_gates(strategy)
    annotated_strategy = strategy.copy()
    LogicalAnnotator(mapping)(annotated_strategy)

    new_moments = [] # type: List[ops.Moment]
    acquaintance_opps = set() # type: Set[FrozenSet[int]]
    for moment in annotated_strategy:
        new_moment = [] # type: List[ops.Operation]
        for op in moment:
            if isinstance(op, AcquaintanceOperation):
                opp = frozenset(cast(Sequence[int], op.logical_indices))
                if opp not in acquaintance_opps:
                    acquaintance_opps.add(opp)
                    new_moment.append(ACQUAINT(*op.qubits))
            else:
                new_moment.append(op)
        new_moments.append(ops.Moment(new_moment))
    strategy._moments = new_moments
