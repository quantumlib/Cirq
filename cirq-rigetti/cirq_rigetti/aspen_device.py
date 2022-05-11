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
from typing import List, cast, Optional, Union, Dict, Any
import functools
from math import sqrt
import httpx
import numpy as np
import networkx as nx
import cirq
from pyquil.quantum_processor import QCSQuantumProcessor
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client


class UnsupportedQubit(ValueError):
    pass


class UnsupportedRigettiQCSOperation(ValueError):
    pass


class UnsupportedRigettiQCSQuantumProcessor(ValueError):
    pass


_grid_qubit_mapping = {
    cirq.GridQubit(0, 0): 2,
    cirq.GridQubit(0, 1): 15,
    cirq.GridQubit(1, 0): 1,
    cirq.GridQubit(1, 1): 16,
}
_forward_line_qubit_mapping = [5, 4, 3, 2]
_reverse_line_qubit_mapping = [1, 0, 7, 6]


@cirq.value.value_equality
class RigettiQCSAspenDevice(cirq.devices.Device):
    """A cirq.Qid supporting Rigetti QCS Aspen device topology."""

    def __init__(self, isa: Union[InstructionSetArchitecture, Dict[str, Any]]) -> None:
        """Initializes a RigettiQCSAspenDevice with its Rigetti QCS `InstructionSetArchitecture`.

        Args:
            isa: The `InstructionSetArchitecture` retrieved from the QCS api.

        Raises:
            UnsupportedRigettiQCSQuantumProcessor: If the isa does not define
            an Aspen device.
        """
        if isinstance(isa, InstructionSetArchitecture):
            self.isa = isa
        else:
            self.isa = InstructionSetArchitecture.from_dict(isa)

        if self.isa.architecture.family.lower() != 'aspen':
            raise UnsupportedRigettiQCSQuantumProcessor(
                'this integration currently only supports Aspen devices, '
                f'but client provided a {self.isa.architecture.family} device'
            )
        self.quantum_processor = QCSQuantumProcessor(
            quantum_processor_id=self.isa.name, isa=self.isa
        )

    def qubits(self) -> List['AspenQubit']:
        """Return list of `AspenQubit`s within device topology.

        Returns:
            List of `AspenQubit`s within device topology.
        """
        qubits = []
        for node in self.isa.architecture.nodes:
            qubits.append(AspenQubit.from_aspen_index(node.node_id))
        return qubits

    @property
    def qubit_topology(self) -> nx.Graph:
        """Return qubit topology indices with nx.Graph.

        Returns:
            Qubit topology as nx.Graph with each node specified with AspenQubit index.
        """
        return self.quantum_processor.qubit_topology()

    @property
    def _number_octagons(self) -> int:
        return int(np.ceil(self._maximum_qubit_number / 10))

    @property
    def _maximum_qubit_number(self) -> int:
        return max([node.node_id for node in self.isa.architecture.nodes])

    @functools.lru_cache(maxsize=2)
    def _line_qubit_mapping(self) -> List[int]:
        mapping: List[int] = []
        for i in range(self._number_octagons):
            base = i * 10
            mapping = mapping + [base + index for index in _forward_line_qubit_mapping]
        for i in range(self._number_octagons):
            base = (self._number_octagons - i - 1) * 10
            mapping = mapping + [base + index for index in _reverse_line_qubit_mapping]
        return mapping

    def _aspen_qubit_index(self, valid_qubit: cirq.Qid) -> int:
        if isinstance(valid_qubit, cirq.GridQubit):
            return _grid_qubit_mapping[valid_qubit]

        if isinstance(valid_qubit, cirq.LineQubit):
            return self._line_qubit_mapping()[valid_qubit.x]

        if isinstance(valid_qubit, cirq.NamedQubit):
            return int(valid_qubit.name)

        if isinstance(valid_qubit, (OctagonalQubit, AspenQubit)):
            return valid_qubit.index

        else:
            # coverage: ignore
            raise UnsupportedQubit(f'unsupported Qid type {type(valid_qubit)}')

    def validate_qubit(self, qubit: 'cirq.Qid') -> None:
        """Raises an exception if the qubit does not satisfy the topological constraints
        of the RigettiQCSAspenDevice.

        Args:
            qubit: The qubit to validate.

        Raises:
            UnsupportedQubit: The operation isn't valid for this device.
        """
        if isinstance(qubit, cirq.GridQubit):
            if self._number_octagons < 2:
                raise UnsupportedQubit('this device does not support GridQubits')
            if not (qubit.row <= 1 and qubit.col <= 1):
                raise UnsupportedQubit(
                    'Aspen devices only support square grids of 1 row and 1 column'
                )
            return

        if isinstance(qubit, cirq.LineQubit):
            if not (qubit.x <= self._number_octagons * 8):
                raise UnsupportedQubit(
                    'this Aspen device only supports line ',
                    f'qubits up to length {self._number_octagons * 8}',
                )
            return

        if isinstance(qubit, cirq.NamedQubit):
            try:
                index = int(qubit.name)
                if not (index < self._maximum_qubit_number):
                    raise UnsupportedQubit(
                        'this Aspen device only supports qubits up to index '
                        f'{self._maximum_qubit_number}'
                    )
                if not ((index % 10) <= 7):
                    raise UnsupportedQubit(
                        'this Aspen device only supports qubit indices mod 10 <= 7'
                    )
                return

            except ValueError:
                raise UnsupportedQubit('Aspen devices only support named qubits by octagonal index')

        if isinstance(qubit, (OctagonalQubit, AspenQubit)):
            if not (qubit.index < self._maximum_qubit_number):
                raise UnsupportedQubit(
                    'this Aspen device only supports ',
                    f'qubits up to index {self._maximum_qubit_number}',
                )
            return

        else:
            # coverage: ignore
            raise UnsupportedQubit(f'unsupported Qid type {type(qubit)}')

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        """Raises an exception if an operation does not satisfy the topological constraints
        of the device.

        Note, in case the operation is invalid, you can still use the Quil
        compiler to rewire qubits and decompose the operation to this device's
        topology.

        Additionally, this method will not attempt to decompose the operation into this
        device's native gate set. This integration, by default, uses the Quil
        compiler to do so.

        Please see the Quil Compiler
        [documentation](https://pyquil-docs.rigetti.com/en/stable/compiler.html)
        for more information.

        Args:
            operation: The operation to validate.

        Raises:
            UnsupportedRigettiQCSOperation: The operation isn't valid for this device.
        """
        qubits = operation.qubits
        for qubit in qubits:
            self.validate_qubit(qubit)
        if len(qubits) == 2:
            i = self._aspen_qubit_index(qubits[0])
            j = self._aspen_qubit_index(qubits[1])
            if j not in self.qubit_topology[i]:
                raise UnsupportedRigettiQCSOperation(
                    f'qubits {qubits[0]} and {qubits[1]} do not share an edge'
                )

    def _value_equality_values_(self):
        return self._maximum_qubit_number

    def __repr__(self):
        return f'cirq_rigetti.RigettiQCSAspenDevice(isa={self.isa!r})'

    def _json_dict_(self):
        return {'isa': self.isa.to_dict()}

    @classmethod
    def _from_json_dict_(cls, isa, **kwargs):
        return cls(isa=InstructionSetArchitecture.from_dict(isa))


@_provide_default_client
def get_rigetti_qcs_aspen_device(
    quantum_processor_id: str, client: Optional[httpx.Client]
) -> RigettiQCSAspenDevice:
    """Retrieves a `qcs_api_client.models.InstructionSetArchitecture` from the Rigetti
    QCS API and uses it to initialize a RigettiQCSAspenDevice.

    Args:
        quantum_processor_id: The identifier of the Rigetti QCS quantum processor.
        client: Optional; A `httpx.Client` initialized with Rigetti QCS credentials
        and configuration. If not provided, `qcs_api_client` will initialize a
        configured client based on configured values in the current user's
        `~/.qcs` directory or default values.

    Returns:
        A `RigettiQCSAspenDevice` with the specified quantum processor instruction
        set and architecture.

    """
    # coverage: ignore
    isa = cast(
        InstructionSetArchitecture,
        get_instruction_set_architecture(
            client=client, quantum_processor_id=quantum_processor_id
        ).parsed,
    )
    return RigettiQCSAspenDevice(isa=isa)


class OctagonalQubit(cirq.ops.Qid):
    """A cirq.Qid supporting Octagonal indexing."""

    def __init__(self, octagon_position: int):
        r"""Initializes an `OctagonalQubit` using indices 0-7.
              4  - 3
            /        \
          5           2
          |           |
          6           1
            \       /
              7 - 0

        Args:
            octagon_position: Position within octagon, indexed as pictured above.

        Returns:
            The initialized `OctagonalQubit`.

        Raises:
            ValueError: If the position specified is greater than 7.
        """
        if octagon_position >= 8:
            raise ValueError(f'OctagonQubit must be less than 8, received {octagon_position}')

        self._octagon_position = octagon_position
        self.index = octagon_position

    @property
    def octagon_position(self):
        return self._octagon_position

    def _comparison_key(self):
        return self.index

    @property
    def dimension(self) -> int:
        return 2

    def distance(self, other: cirq.Qid) -> float:
        """Returns the distance between two qubits.

        Args:
            other: An OctagonalQubit to which we are measuring distance.

        Returns:
            The distance between two qubits.

        Raises:
            TypeError: other qubit must be OctagonalQubit.
        """
        if type(other) != OctagonalQubit:
            raise TypeError("can only measure distance from other Octagonal qubits")
        other = cast(OctagonalQubit, other)
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    @property
    def x(self) -> float:
        """Returns the horizontal position of the qubit, assuming each side of
        the octagon has length 1.

        Returns:
            The horizontal position of the qubit.

        Raises:
            ValueError: Octagon position is invalid.
        """
        if self.octagon_position in {5, 6}:
            return 0
        if self.octagon_position in {4, 7}:
            return 1 / sqrt(2)
        if self.octagon_position in {0, 3}:
            return 1 + 1 / sqrt(2)
        if self.octagon_position in {1, 2}:
            return 1 + sqrt(2)

        raise ValueError(f'invalid octagon position {self.octagon_position}')

    @property
    def y(self) -> float:
        """Returns the vertical position of the qubit, assuming each side of
        the octagon has length 1. The y-axis is oriented downwards.

        Returns:
            The vertical position of the qubit.

        Raises:
            ValueError: Octagon position is invalid.
        """
        if self.octagon_position in {3, 4}:
            return 0
        if self.octagon_position in {2, 5}:
            return 1 / sqrt(2)
        if self.octagon_position in {1, 6}:
            return 1 + 1 / sqrt(2)
        if self.octagon_position in {0, 7}:
            return 1 + sqrt(2)

        raise ValueError(f'invalid octagon position {self.octagon_position}')

    @property
    def z(self) -> int:
        """Because this is a 2-dimensional qubit, this will always be 0.

        Returns:
            Zero.
        """
        return 0

    def __repr__(self):
        return f'cirq_rigetti.OctagonalQubit(octagon_position={self.octagon_position})'

    def _json_dict_(self):
        return {'octagon_position': self.octagon_position}


class AspenQubit(OctagonalQubit):
    def __init__(self, octagon: int, octagon_position: int):
        super(AspenQubit, self).__init__(octagon_position)
        self._octagon = octagon
        self.index = octagon * 10 + octagon_position

    @property
    def octagon(self):
        return self._octagon

    def _comparison_key(self):
        return self.octagon, self.index

    @property
    def x(self) -> float:
        """Returns the horizontal position of the qubit, assuming each side of
        the octagon has length 1.

        Returns:
            The horizontal position of the qubit.

        Raises:
            ValueError: Octagon position is invalid.
        """
        octagon_left_most_position = self.octagon * (2 + sqrt(2))
        if self.octagon_position in {5, 6}:
            return octagon_left_most_position
        if self.octagon_position in {4, 7}:
            return octagon_left_most_position + 1 / sqrt(2)
        if self.octagon_position in {0, 3}:
            return octagon_left_most_position + 1 + 1 / sqrt(2)
        if self.octagon_position in {1, 2}:
            return octagon_left_most_position + 1 + sqrt(2)

        raise ValueError(f'invalid octagon position {self.octagon_position}')

    def distance(self, other: cirq.Qid) -> float:
        """Returns the distance between two qubits.

        Args:
            other: An AspenQubit to which we are measuring distance.

        Returns:
            The distance between two qubits.
        Raises:
            TypeError: other qubit must be AspenQubit.
        """
        if type(other) != AspenQubit:
            raise TypeError("can only measure distance from other Aspen qubits")
        other = cast(AspenQubit, other)
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def to_grid_qubit(self) -> cirq.GridQubit:
        """Converts `AspenQubit` to `cirq.GridQubit`.

        Returns:
            The equivalent GridQubit.

        Raises:
            ValueError: AspenQubit cannot be converted to GridQubit.
        """
        for grid_qubit, aspen_index in _grid_qubit_mapping.items():
            if self.index == aspen_index:
                return grid_qubit
        raise ValueError(f'cannot use {self} as a GridQubit')

    def to_named_qubit(self) -> cirq.NamedQubit:
        """Converts `AspenQubit` to `cirq.NamedQubit`.

        Returns:
            The equivalent NamedQubit.
        """
        return cirq.NamedQubit(str(self.index))

    @staticmethod
    def from_grid_qubit(grid_qubit: cirq.GridQubit) -> 'AspenQubit':
        """Converts `cirq.GridQubit` to `AspenQubit`.

        Returns:
            The equivalent AspenQubit.

        Raises:
            ValueError: GridQubit cannot be converted to AspenQubit.
        """
        if grid_qubit in _grid_qubit_mapping:
            return AspenQubit.from_aspen_index(_grid_qubit_mapping[grid_qubit])
        raise ValueError(f'{grid_qubit} is not convertible to Aspen qubit')

    @staticmethod
    def from_named_qubit(qubit: cirq.NamedQubit) -> 'AspenQubit':
        """Converts `cirq.NamedQubit` to `AspenQubit`.

        Returns:
            The equivalent AspenQubit.

        Raises:
            ValueError: NamedQubit cannot be converted to AspenQubit.
            UnsupportedQubit: If the supplied qubit is not a named qubit with an octagonal
                index.
        """
        try:
            index = int(qubit.name)
            return AspenQubit.from_aspen_index(index)
        except ValueError:
            raise UnsupportedQubit('Aspen devices only support named qubits by octagonal index')

    @staticmethod
    def from_aspen_index(index: int) -> 'AspenQubit':
        """Initializes an `AspenQubit` at the given index. See `OctagonalQubit` to understand
        OctagonalQubit indexing.

        Args:
            index: The index at which to initialize the `AspenQubit`.

        Returns:
            The AspenQubit with requested index.

        Raises:
            ValueError: index is not a valid octagon position.
        """
        octagon_position = index % 10
        octagon = np.floor(index / 10.0)
        return AspenQubit(octagon, octagon_position)

    def __repr__(self):
        return (
            f'cirq_rigetti.AspenQubit('
            f'octagon={self.octagon}, octagon_position={self.octagon_position})'
        )

    def __str__(self):
        return f'({self.octagon}, {self.octagon_position})'

    def _json_dict_(self):
        return {'octagon': self.octagon, 'octagon_position': self.octagon_position}
