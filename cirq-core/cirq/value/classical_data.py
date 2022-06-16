# Copyright 2022 The Cirq Developers
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

import abc
import enum
from typing import Dict, List, Mapping, Sequence, Tuple, TYPE_CHECKING, TypeVar

from cirq.value import digits, value_equality_attr

if TYPE_CHECKING:
    import cirq


class MeasurementType(enum.IntEnum):
    """Type of a measurement, whether a measurement or channel.

    This determines how the results of a measurement are stored
    as classical data in a `ClassicalDataStoreRegister`.
    `MEASUREMENT` represent measurements of a `Cirq.Qid`
    (for instance, a qubit or qudit).  A `CHANNEL` represents
    the measurement of a channel, such as the set of Kraus
    operators.  In this case, the data stored is the integer
    index of the channel measured.
    """

    MEASUREMENT = 1
    CHANNEL = 2

    def __repr__(self):
        return f'cirq.{str(self)}'


TSelf = TypeVar('TSelf', bound='ClassicalDataStoreReader')


class ClassicalDataStoreReader(abc.ABC):
    @abc.abstractmethod
    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        """Gets the measurement keys in the order they were stored."""

    @property
    @abc.abstractmethod
    def records(self) -> Mapping['cirq.MeasurementKey', List[Tuple[int, ...]]]:
        """Gets the a mapping from measurement key to measurement records."""

    @property
    @abc.abstractmethod
    def channel_records(self) -> Mapping['cirq.MeasurementKey', List[int]]:
        """Gets the a mapping from measurement key to channel measurement records."""

    @abc.abstractmethod
    def get_int(self, key: 'cirq.MeasurementKey', index=-1) -> int:
        """Gets the integer corresponding to the measurement.

        The integer is determined by summing the qubit-dimensional basis value
        of each measured qubit. For example, if the measurement of qubits
        [q1, q0] produces [1, 0], then the corresponding integer is 2, the big-
        endian equivalent. If they are qutrits and the measurement is [2, 1],
        then the integer is 2 * 3 + 1 = 7.

        Args:
            key: The measurement key.
            index: If multiple measurements have the same key, the index
                argument can be used to specify which measurement to retrieve.
                Here `0` refers to the first measurement, and `-1` refers to
                the most recent.

        Raises:
            KeyError: If the key has not been used or if the index is out of
                bounds.
        """

    @abc.abstractmethod
    def get_digits(self, key: 'cirq.MeasurementKey', index=-1) -> Tuple[int, ...]:
        """Gets the values of the qubits that were measured into this key.

        For example, if the measurement of qubits [q0, q1] produces [0, 1],
        this function will return (0, 1).

        Args:
            key: The measurement key.
            index: If multiple measurements have the same key, the index
                argument can be used to specify which measurement to retrieve.
                Here `0` refers to the first measurement, and `-1` refers to
                the most recent.

        Raises:
            KeyError: If the key has not been used or if the index is out of
                bounds.
        """

    @abc.abstractmethod
    def copy(self: TSelf) -> TSelf:
        """Creates a copy of the object."""


class ClassicalDataStore(ClassicalDataStoreReader, abc.ABC):
    @abc.abstractmethod
    def record_measurement(
        self, key: 'cirq.MeasurementKey', measurement: Sequence[int], qubits: Sequence['cirq.Qid']
    ):
        """Records a measurement.

        Args:
            key: The measurement key to hold the measurement.
            measurement: The measurement result.
            qubits: The qubits that were measured.

        Raises:
            ValueError: If the measurement shape does not match the qubits
                measured or if the measurement key was already used.
        """

    @abc.abstractmethod
    def record_channel_measurement(self, key: 'cirq.MeasurementKey', measurement: int):
        """Records a channel measurement.

        Args:
            key: The measurement key to hold the measurement.
            measurement: The measurement result.

        Raises:
            ValueError: If the measurement key was already used.
        """


@value_equality_attr.value_equality(unhashable=True)
class ClassicalDataDictionaryStore(ClassicalDataStore):
    """Classical data representing measurements and metadata."""

    def __init__(
        self,
        *,
        _records: Dict['cirq.MeasurementKey', List[Tuple[int, ...]]] = None,
        _measured_qubits: Dict['cirq.MeasurementKey', List[Tuple['cirq.Qid', ...]]] = None,
        _channel_records: Dict['cirq.MeasurementKey', List[int]] = None,
        _measurement_types: Dict['cirq.MeasurementKey', 'cirq.MeasurementType'] = None,
    ):
        """Initializes a `ClassicalDataDictionaryStore` object."""
        if not _measurement_types:
            _measurement_types = {}
            if _records:
                _measurement_types.update(
                    {k: MeasurementType.MEASUREMENT for k, v in _records.items()}
                )
            if _channel_records:
                _measurement_types.update(
                    {k: MeasurementType.CHANNEL for k, v in _channel_records.items()}
                )
        if _records is None:
            _records = {}
        if _measured_qubits is None:
            _measured_qubits = {}
        if _channel_records is None:
            _channel_records = {}
        self._records: Dict['cirq.MeasurementKey', List[Tuple[int, ...]]] = _records
        self._measured_qubits: Dict[
            'cirq.MeasurementKey', List[Tuple['cirq.Qid', ...]]
        ] = _measured_qubits
        self._channel_records: Dict['cirq.MeasurementKey', List[int]] = _channel_records
        self._measurement_types: Dict[
            'cirq.MeasurementKey', 'cirq.MeasurementType'
        ] = _measurement_types

    @property
    def records(self) -> Mapping['cirq.MeasurementKey', List[Tuple[int, ...]]]:
        """Gets the a mapping from measurement key to measurement records."""
        return self._records

    @property
    def channel_records(self) -> Mapping['cirq.MeasurementKey', List[int]]:
        """Gets the a mapping from measurement key to channel measurement records."""
        return self._channel_records

    @property
    def measured_qubits(self) -> Mapping['cirq.MeasurementKey', List[Tuple['cirq.Qid', ...]]]:
        """Gets the a mapping from measurement key to the qubits measured."""
        return self._measured_qubits

    @property
    def measurement_types(self) -> Mapping['cirq.MeasurementKey', 'cirq.MeasurementType']:
        """Gets the a mapping from measurement key to the measurement type."""
        return self._measurement_types

    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        return tuple(self._measurement_types.keys())

    def record_measurement(
        self, key: 'cirq.MeasurementKey', measurement: Sequence[int], qubits: Sequence['cirq.Qid']
    ):
        if len(measurement) != len(qubits):
            raise ValueError(f'{len(measurement)} measurements but {len(qubits)} qubits.')
        if key not in self._measurement_types:
            self._measurement_types[key] = MeasurementType.MEASUREMENT
            self._records[key] = []
            self._measured_qubits[key] = []
        if self._measurement_types[key] != MeasurementType.MEASUREMENT:
            raise ValueError(f"Channel Measurement already logged to key {key}")
        measured_qubits = self._measured_qubits[key]
        if measured_qubits:
            shape = tuple(q.dimension for q in qubits)
            key_shape = tuple(q.dimension for q in measured_qubits[-1])
            if shape != key_shape:
                raise ValueError(f'Measurement shape {shape} does not match {key_shape} in {key}.')
        measured_qubits.append(tuple(qubits))
        self._records[key].append(tuple(measurement))

    def record_channel_measurement(self, key: 'cirq.MeasurementKey', measurement: int):
        if key not in self._measurement_types:
            self._measurement_types[key] = MeasurementType.CHANNEL
            self._channel_records[key] = []
        if self._measurement_types[key] != MeasurementType.CHANNEL:
            raise ValueError(f"Measurement already logged to key {key}")
        self._channel_records[key].append(measurement)

    def get_digits(self, key: 'cirq.MeasurementKey', index=-1) -> Tuple[int, ...]:
        return (
            self._records[key][index]
            if self._measurement_types[key] == MeasurementType.MEASUREMENT
            else (self._channel_records[key][index],)
        )

    def get_int(self, key: 'cirq.MeasurementKey', index=-1) -> int:
        if key not in self._measurement_types:
            raise KeyError(f'The measurement key {key} is not in {self._measurement_types}')
        measurement_type = self._measurement_types[key]
        if measurement_type == MeasurementType.CHANNEL:
            return self._channel_records[key][index]
        if key not in self._measured_qubits:
            return digits.big_endian_bits_to_int(self._records[key][index])
        return digits.big_endian_digits_to_int(
            self._records[key][index], base=[q.dimension for q in self._measured_qubits[key][index]]
        )

    def copy(self):
        return ClassicalDataDictionaryStore(
            _records=self._records.copy(),
            _measured_qubits=self._measured_qubits.copy(),
            _channel_records=self._channel_records.copy(),
            _measurement_types=self._measurement_types.copy(),
        )

    def _json_dict_(self):
        return {
            'records': list(self.records.items()),
            'measured_qubits': list(self.measured_qubits.items()),
            'channel_records': list(self.channel_records.items()),
            'measurement_types': list(self.measurement_types.items()),
        }

    @classmethod
    def _from_json_dict_(
        cls, records, measured_qubits, channel_records, measurement_types, **kwargs
    ):
        return cls(
            _records=dict(records),
            _measured_qubits=dict(measured_qubits),
            _channel_records=dict(channel_records),
            _measurement_types=dict(measurement_types),
        )

    def __repr__(self):
        rep = 'cirq.ClassicalDataDictionaryStore('
        if self.records:
            rep += f'_records={self.records!r},'
        if self.measured_qubits:
            rep += f' _measured_qubits={self.measured_qubits!r},'
        if self.channel_records:
            rep += f' _channel_records={self.channel_records!r},'
        if self.measurement_types:
            rep += f' _measurement_types={self.measurement_types!r},'
        return rep + ')'

    def _value_equality_values_(self):
        return (
            self._records,
            self._channel_records,
            self._measurement_types,
            self._measured_qubits,
        )
