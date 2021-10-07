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
"""An interface for quantum states as targets for operations."""
import abc
from typing import (
    TypeVar,
    TYPE_CHECKING,
    Generic,
    Dict,
    Any,
    Tuple,
    Optional,
    Iterator,
    List,
    Sequence,
    Union,
)

import numpy as np

from cirq import protocols
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


TSelfTarget = TypeVar('TSelfTarget', bound='OperationTarget')
TActOnArgs = TypeVar('TActOnArgs', bound='cirq.ActOnArgs')


class OperationTarget(Generic[TActOnArgs], metaclass=abc.ABCMeta):
    """An interface for quantum states as targets for operations."""

    @abc.abstractmethod
    def create_merged_state(self) -> TActOnArgs:
        """Creates a final merged state."""

    @abc.abstractmethod
    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> Union[bool, NotImplementedType]:
        """Handles the act_on protocol fallback implementation.

        Args:
            action: Either a gate or an operation to act on.
            qubits: The applicable qubits if a gate is passed as the action.
            allow_decompose: Flag to allow decomposition.

        Returns:
            True if the fallback applies, else NotImplemented."""

    def apply_operation(self, op: 'cirq.Operation'):
        protocols.act_on(op, self)

    @abc.abstractmethod
    def copy(self: TSelfTarget) -> TSelfTarget:
        """Copies the object."""

    @property
    @abc.abstractmethod
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        """Gets the qubit order maintained by this target."""

    @property
    @abc.abstractmethod
    def log_of_measurement_results(self) -> Dict[str, Any]:
        """Gets the log of measurement results."""

    @abc.abstractmethod
    def sample(
        self,
        qubits: List['cirq.Qid'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        """Samples the state value."""

    def __getitem__(self, item: Optional['cirq.Qid']) -> TActOnArgs:
        """Gets the item associated with the qubit."""

    def __len__(self) -> int:
        """Gets the number of items in the mapping."""

    def __iter__(self) -> Iterator[Optional['cirq.Qid']]:
        """Iterates the keys of the mapping."""
