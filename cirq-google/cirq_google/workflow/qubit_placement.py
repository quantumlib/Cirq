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
from typing import Dict, Any, Tuple, TYPE_CHECKING

import numpy as np

import cirq
from cirq import _compat

if TYPE_CHECKING:
    import cirq_google as cg


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
