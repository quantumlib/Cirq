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

from __future__ import annotations

import operator
import random
from typing import Any, Callable, cast, Iterable, TYPE_CHECKING

import networkx

from cirq import ops

if TYPE_CHECKING:
    import cirq


def is_topologically_sorted(
    dag: cirq.contrib.CircuitDag,
    operations: cirq.OP_TREE,
    equals: Callable[[ops.Operation, ops.Operation], bool] = operator.eq,
) -> bool:
    """Whether a given order of operations is consistent with the DAG.

    For example, suppose the (transitive reduction of the) circuit DAG is

         ╭─> Op2 ─╮
    Op1 ─┤        ├─> Op4
         ╰─> Op3 ─╯

    Then [Op1, Op2, Op3, Op4] and [Op1, Op3, Op2, Op4] (and any operations
    tree that flattens to one of them) are topologically sorted according
    to the DAG, and any other ordering of the four operations is not.

    Evaluates to False when the set of operations is different from those
    in the nodes of the DAG, regardless of the ordering.

    Args:
        dag: The circuit DAG.
        operations: The ordered operations.
        equals: The function to determine equality of operations. Defaults to
            `operator.eq`.

    Returns:
        Whether or not the operations given are topologically sorted
        according to the DAG.
    """

    remaining_dag = dag.copy()
    frontier = [node for node in remaining_dag.nodes() if not remaining_dag.pred[node]]
    for operation in cast(Iterable[ops.Operation], ops.flatten_op_tree(operations)):
        for i, node in enumerate(frontier):
            if equals(node.val, operation):
                frontier.pop(i)
                succ = remaining_dag.succ[node]
                remaining_dag.remove_node(node)
                frontier.extend(new_node for new_node in succ if not remaining_dag.pred[new_node])
                break
        else:
            return False
    return not bool(frontier)


def random_topological_sort(dag: networkx.DiGraph) -> Iterable[Any]:
    remaining_dag = dag.copy()
    frontier = list(node for node in remaining_dag.nodes() if not remaining_dag.pred[node])
    while frontier:
        random.shuffle(frontier)
        node = frontier.pop()
        succ = remaining_dag.succ[node]
        remaining_dag.remove_node(node)
        frontier.extend(new_node for new_node in succ if not remaining_dag.pred[new_node])
        yield node
    assert not remaining_dag
