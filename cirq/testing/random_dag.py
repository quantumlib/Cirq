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

import random
from typing import Any, Iterable

import networkx

from cirq import circuits
from cirq.testing.random_circuit import random_circuit

def random_circuit_dag(*args, **kwargs) -> circuits.Circuit:
    """Generates a random circuit DAG.

    Args:
        *args: The arguments to pass to random_circuit.
        **kwargs: The keyword arguments to pass to random_circuit.

    Raises:
        ValueError: random_circuit raises a ValueError on the given arguments.

    Returns:
        The randomly generated CircuitDag.
    """

    return circuits.CircuitDag.from_circuit(random_circuit(*args, **kwargs))


def random_topological_sort(dag: networkx.DiGraph) -> Iterable[Any]:
    remaining_dag = dag.copy()
    frontier = list(node for node in remaining_dag.nodes()
                   if not remaining_dag.pred[node])
    while frontier:
        random.shuffle(frontier)
        node = frontier.pop()
        succ = remaining_dag.succ[node]
        remaining_dag.remove_node(node)
        frontier.extend(new_node for new_node in succ
                        if not remaining_dag.pred[new_node])
        yield node
    assert not remaining_dag
