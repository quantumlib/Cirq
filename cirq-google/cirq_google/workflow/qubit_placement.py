# Copyright 2021 The Cirq Developers
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

"""Features for placing qubits onto devices."""

import abc
import dataclasses
from functools import lru_cache
from typing import Dict, Any, Tuple, TYPE_CHECKING
from typing import List, Callable

import numpy as np

import cirq
import cirq_google as cg
from cirq import _compat
from cirq.devices.named_topologies import get_placements, NamedTopology
from cirq.protocols import obj_to_dict_helper

if TYPE_CHECKING:
    import cirq_google as cg


class CouldNotPlaceError(RuntimeError):
    """Raised if a problem topology could not be placed on a device graph."""


def default_topo_node_to_qubit(node: Any) -> cirq.Qid:
    try:
        return cirq.GridQubit(*node)
    except TypeError:
        return cirq.LineQubit(node)


class QubitPlacer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def place_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        problem_topology: 'cirq.NamedTopology',
        shared_rt_info: 'cg.SharedRuntimeInfo',
        rs: np.random.RandomState,
    ) -> Tuple['cirq.FrozenCircuit', Dict[Any, 'cirq.Qid']]:
        """Place a circuit with a given topology.

        Args:
            circuit: The circuit.
            problem_topology: The topologies (i.e. connectivity) of the circuit.
            shared_rt_info: A `cg.SharedRuntimeInfo` object that may contain additional info
                to inform placement.
            rs: A `RandomState` to enable pseudo-random placement strategies.

        Returns:
            A tuple of a new frozen circuit with the qubits placed and a mapping from input
            qubits or nodes to output qubits.
        """


@dataclasses.dataclass(frozen=True)
class NaiveQubitPlacer(QubitPlacer):
    """Don't do any qubit placement, use circuit qubits."""

    def place_circuit(
        self,
        circuit: 'cirq.AbstractCircuit',
        problem_topology: 'cirq.NamedTopology',
        shared_rt_info: 'cg.SharedRuntimeInfo',
        rs: np.random.RandomState,
    ) -> Tuple['cirq.FrozenCircuit', Dict[Any, 'cirq.Qid']]:
        return circuit.freeze(), {q: q for q in circuit.all_qubits()}

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


class OffsetQubitPlacer(QubitPlacer):
    def __init__(self, offset=(0, 0)):
        self.offset = offset

    def place_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        problem_topology: NamedTopology,
        shared_rt_info: 'cg.SharedRuntimeInfo',
        rs: np.random.RandomState,
    ) -> Tuple[cirq.FrozenCircuit, Dict[Any, cirq.Qid]]:
        mapping = {q: q + self.offset for q in circuit.all_qubits()}
        circuit = circuit.unfreeze().transform_qubits(mapping).freeze()
        return circuit, mapping

    def _json_dict_(self):
        return obj_to_dict_helper(self, attribute_names=[], namespace='recirq.google')


def _zigzag(offset, length):
    up = True
    nodes = [cirq.GridQubit(*offset)]
    for i in range(1, length):
        if up:
            nodes.append(nodes[-1] + (0, 1))
        else:
            nodes.append(nodes[-1] + (1, 0))
        up = not up
    return nodes


class HardcodedQubitPlacer(QubitPlacer):
    def __init__(
        self,
        processor_id: str,
        handpick_index: int,
        topo_node_to_qubit_func: Callable[[Any], cirq.Qid] = default_topo_node_to_qubit,
    ):
        self.processor_id = processor_id
        self.handpick_index = handpick_index
        self.topo_node_to_qubit_func = topo_node_to_qubit_func

    def place_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        problem_topology: NamedTopology,
        shared_rt_info: 'cg.SharedRuntimeInfo',
        rs: np.random.RandomState,
    ) -> Tuple[cirq.FrozenCircuit, Dict[Any, cirq.Qid]]:

        if not isinstance(problem_topology, cirq.TiltedSquareLattice):
            raise CouldNotPlaceError()

        topos = [
            cirq.TiltedSquareLattice(2, 2),
            cirq.TiltedSquareLattice(4, 2),
            cirq.LineTopology(5),
        ]

        if problem_topology not in topos:
            raise CouldNotPlaceError()

        hardcoded_map = {
            'rainbow': [
                {
                    topos[0]: topos[0].nodes_to_gridqubits(offset=(3, 2)),
                    topos[1]: topos[1].nodes_to_gridqubits(offset=(3, 2)),
                    topos[2]: _zigzag(offset=(4, 1), length=5),
                },
                {
                    topos[0]: topos[0].nodes_to_gridqubits(offset=(5, 2)),
                    topos[1]: topos[1].nodes_to_gridqubits(offset=(5, 2)),
                    topos[2]: _zigzag(offset=(4, 2), length=5),
                },
            ],
            'weber': [
                {
                    topos[0]: topos[0].nodes_to_gridqubits(offset=(1, 6)),
                    topos[1]: topos[1].nodes_to_gridqubits(offset=(1, 6)),
                    topos[2]: {i: cirq.GridQubit(4, i + 2) for i in range(5)},
                },
                {
                    topos[0]: topos[0].nodes_to_gridqubits(offset=(4, 4)),
                    topos[1]: topos[1].nodes_to_gridqubits(offset=(4, 4)),
                    topos[2]: {i: cirq.GridQubit(5, i + 2) for i in range(5)},
                },
            ],
        }

        nt_mapping = hardcoded_map[self.processor_id][self.handpick_index][problem_topology]
        # NamedTopologies let us build a mapping from [node] to GridQubit where
        # [node] is an int or tuple of ints depending on the topology type.
        circuit_mapping = {
            self.topo_node_to_qubit_func(tsl_node): gridq for tsl_node, gridq in nt_mapping.items()
        }

        circuit = circuit.unfreeze().transform_qubits(circuit_mapping).freeze()
        return circuit, circuit_mapping

    def _json_dict_(self):
        return obj_to_dict_helper(self, attribute_names=[], namespace='recirq.google')


@lru_cache()
def cached_get_placements(
    problem_topo: NamedTopology, device: 'cirq.Device'
) -> List[Dict[Any, cirq.GridQubit]]:
    """Cache placements onto the specific device."""
    return get_placements(big_graph=device.get_nx_graph(), small_graph=problem_topo.graph)


def get_random_placement(
    problem_topo: NamedTopology,
    device: 'cirq.Device',
    rs: np.random.RandomState,
    topo_node_to_qubit_func: Callable[[Any], cirq.Qid] = default_topo_node_to_qubit,
) -> Dict[Any, cirq.GridQubit]:
    """Place `problem_topology` randomly onto a device."""
    placements = cached_get_placements(problem_topo, device)
    if len(placements) == 0:
        raise CouldNotPlaceError
    random_i = int(rs.random_integers(0, len(placements) - 1, size=1))
    placement = placements[random_i]
    placement_gq = {topo_node_to_qubit_func(k): v for k, v in placement.items()}
    return placement_gq


class RandomDevicePlacer(QubitPlacer):
    def __init__(
        self,
        topo_node_to_qubit_func: Callable[[Any], cirq.Qid] = default_topo_node_to_qubit,
    ):
        self.topo_node_to_qubit_func = topo_node_to_qubit_func

    def place_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        problem_topology: NamedTopology,
        shared_rt_info: 'cg.SharedRuntimeInfo',
        rs: np.random.RandomState,
    ) -> Tuple[cirq.FrozenCircuit, Dict[Any, cirq.Qid]]:
        device = shared_rt_info.device
        placement = get_random_placement(
            problem_topology, device, rs=rs, topo_node_to_qubit_func=self.topo_node_to_qubit_func
        )
        return circuit.unfreeze().transform_qubits(placement).freeze(), placement
