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
from typing import Any, Callable, Dict, Hashable, List, Tuple, TYPE_CHECKING

import numpy as np

import cirq
from cirq import _compat
from cirq.devices.named_topologies import get_placements, NamedTopology
from cirq.protocols import obj_to_dict_helper
from cirq_google.workflow._device_shim import _Device_dot_get_nx_graph

if TYPE_CHECKING:
    import cirq_google as cg


class CouldNotPlaceError(RuntimeError):
    """Raised if a problem topology could not be placed on a device graph."""


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


def default_topo_node_to_qubit(node: Any) -> cirq.Qid:
    """The default mapping from `cirq.NamedTopology` nodes and `cirq.Qid`.

    There is a correspondence between nodes and the "abstract" Qids
    used to construct un-placed circuit. `cirq.get_placements` returns a dictionary
    mapping from node to Qid. We use this function to transform it into a mapping
    from "abstract" Qid to device Qid. This function encodes the default behavior used by
    `RandomDevicePlacer`.

    If nodes are tuples of integers, map to `cirq.GridQubit`. Otherwise, try
    to map to `cirq.LineQubit` and rely on its validation.

    Args:
        node: A node from a `cirq.NamedTopology` graph.

    Returns:
        A `cirq.Qid` appropriate for the node type.
    """

    try:
        return cirq.GridQubit(*node)
    except TypeError:
        return cirq.LineQubit(node)


class HardcodedQubitPlacer(QubitPlacer):
    def __init__(
        self,
        mapping: Dict[cirq.NamedTopology, Dict[Any, cirq.Qid]],
        topo_node_to_qubit_func: Callable[[Hashable], cirq.Qid] = default_topo_node_to_qubit,
    ):
        """A placement strategy that uses the explicitly provided `mapping`.

        Args:
            mapping: The hardcoded placements. This provides a placement for each supported
                `cirq.NamedTopology`. The topology serves as the key for the mapping dictionary.
                Each placement is a dictionary mapping topology node to final `cirq.Qid` device
                qubit.
            topo_node_to_qubit_func: A function that maps from `cirq.NamedTopology` nodes
                to `cirq.Qid`. There is a correspondence between nodes and the "abstract" Qids
                used to construct the un-placed circuit. We use this function to interpret
                the provided mappings. By default: nodes which are tuples correspond
                to `cirq.GridQubit`s; otherwise `cirq.LineQubit`.

        Note:
            The attribute `topo_node_to_qubit_func` is not preserved in JSON serialization. This
            bit of plumbing does not affect the placement behavior.
        """
        self._mapping = mapping
        self.topo_node_to_qubit_func = topo_node_to_qubit_func

    def place_circuit(
        self,
        circuit: cirq.AbstractCircuit,
        problem_topology: NamedTopology,
        shared_rt_info: 'cg.SharedRuntimeInfo',
        rs: np.random.RandomState,
    ) -> Tuple[cirq.FrozenCircuit, Dict[Any, cirq.Qid]]:
        """Place a circuit according to the hardcoded placements.

        Args:
            circuit: The circuit.
            problem_topology: The topologies (i.e. connectivity) of the circuit, use to look
                up the placement in `self.mapping`.
            shared_rt_info: A `cg.SharedRuntimeInfo` object; ignored for hardcoded placement.
            rs: A `RandomState`; ignored for hardcoded placement.

        Returns:
            A tuple of a new frozen circuit with the qubits placed and a mapping from input
            qubits or nodes to output qubits.

        Raises:
            CouldNotPlaceError: if the given problem_topology is not present in the hardcoded
                mapping.
        """
        try:
            nt_mapping = self._mapping[problem_topology]
        except KeyError as e:
            raise CouldNotPlaceError(str(e))

        circuit_mapping = {
            self.topo_node_to_qubit_func(nt_node): gridq for nt_node, gridq in nt_mapping.items()
        }

        circuit = circuit.unfreeze().transform_qubits(circuit_mapping).freeze()
        return circuit, circuit_mapping

    def __repr__(self) -> str:
        return f'cirq_google.HardcodedQubitPlacer(mapping={_compat.proper_repr(self._mapping)})'

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        d = obj_to_dict_helper(self, attribute_names=[])

        # Nested dict: turn both levels to list(key_value_pair)
        mapping = {topo: list(placement.items()) for topo, placement in self._mapping.items()}
        mapping = list(mapping.items())
        d['mapping'] = mapping
        return d

    @classmethod
    def _from_json_dict_(cls, **kwargs) -> 'HardcodedQubitPlacer':
        # From nested list(key_value_pair) to dictionary
        mapping: Dict[cirq.NamedTopology, Dict[Any, 'cirq.Qid']] = {}
        for topo, placement_kvs in kwargs['mapping']:
            placement: Dict[Hashable, 'cirq.Qid'] = {}
            for k, v in placement_kvs:
                if isinstance(k, list):
                    k = tuple(k)
                placement[k] = v
            mapping[topo] = placement

        return cls(mapping=mapping)

    def __eq__(self, other):
        if not isinstance(other, HardcodedQubitPlacer):
            return False  # pragma: no cover

        return self._mapping == other._mapping


@lru_cache()
def _cached_get_placements(
    problem_topo: 'cirq.NamedTopology', device: 'cirq.Device'
) -> List[Dict[Any, 'cirq.Qid']]:
    """Cache `cirq.get_placements` onto the specific device."""
    return get_placements(
        big_graph=_Device_dot_get_nx_graph(device), small_graph=problem_topo.graph
    )


def _get_random_placement(
    problem_topology: 'cirq.NamedTopology',
    device: 'cirq.Device',
    rs: np.random.RandomState,
    topo_node_to_qubit_func: Callable[[Any], 'cirq.Qid'] = default_topo_node_to_qubit,
) -> Dict['cirq.Qid', 'cirq.Qid']:
    """Place `problem_topology` randomly onto a device.

    This is a helper function used by `RandomDevicePlacer.place_circuit`.
    """
    placements = _cached_get_placements(problem_topology, device)
    if len(placements) == 0:
        raise CouldNotPlaceError
    random_i = rs.randint(len(placements))
    placement = placements[random_i]
    placement_gq = {topo_node_to_qubit_func(k): v for k, v in placement.items()}
    return placement_gq


class RandomDevicePlacer(QubitPlacer):
    def __init__(
        self, topo_node_to_qubit_func: Callable[[Any], cirq.Qid] = default_topo_node_to_qubit
    ):
        """A placement strategy that randomly places circuits onto devices.

        Args:
            topo_node_to_qubit_func: A function that maps from `cirq.NamedTopology` nodes
                to `cirq.Qid`. There is a correspondence between nodes and the "abstract" Qids
                used to construct the un-placed circuit. `cirq.get_placements` returns a dictionary
                mapping from node to Qid. We use this function to transform it into a mapping
                from "abstract" Qid to device Qid. By default: nodes which are tuples correspond
                to `cirq.GridQubit`s; otherwise `cirq.LineQubit`.

        Note:
            The attribute `topo_node_to_qubit_func` is not preserved in JSON serialization. This
            bit of plumbing does not affect the placement behavior.
        """
        self.topo_node_to_qubit_func = topo_node_to_qubit_func

    def place_circuit(
        self,
        circuit: 'cirq.AbstractCircuit',
        problem_topology: 'cirq.NamedTopology',
        shared_rt_info: 'cg.SharedRuntimeInfo',
        rs: np.random.RandomState,
    ) -> Tuple['cirq.FrozenCircuit', Dict[Any, 'cirq.Qid']]:
        """Place a circuit with a given topology onto a device via `cirq.get_placements` with
        randomized selection of the placement each time.

        This requires device information to be present in `shared_rt_info`.

        Args:
            circuit: The circuit.
            problem_topology: The topologies (i.e. connectivity) of the circuit.
            shared_rt_info: A `cg.SharedRuntimeInfo` object that contains a `device` attribute
                of type `cirq.Device` to enable placement.
            rs: A `RandomState` as a source of randomness for random placements.

        Returns:
            A tuple of a new frozen circuit with the qubits placed and a mapping from input
            qubits or nodes to output qubits.

        Raises:
            ValueError: If `shared_rt_info` does not have a device field.
        """
        device = shared_rt_info.device
        if device is None:
            raise ValueError(
                "RandomDevicePlacer requires shared_rt_info.device to be a `cirq.Device`. "
                "This should have been set during the initialization phase of `cg.execute`."
            )
        placement = _get_random_placement(
            problem_topology, device, rs=rs, topo_node_to_qubit_func=self.topo_node_to_qubit_func
        )
        return circuit.unfreeze().transform_qubits(placement).freeze(), placement

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, [])

    def __repr__(self) -> str:
        return "cirq_google.RandomDevicePlacer()"

    def __eq__(self, other):
        if isinstance(other, RandomDevicePlacer):
            return True
