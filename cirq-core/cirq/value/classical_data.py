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
from typing import Dict, Mapping, Sequence, Tuple, TYPE_CHECKING, TypeVar

from cirq.value import digits, value_equality_attr

if TYPE_CHECKING:
    import cirq


class MeasurementType(enum.IntEnum):
    MEASUREMENT = 1
    CHANNEL = 2

    def __repr__(self):
        return f'cirq.{str(self)}'


TSelf = TypeVar('TSelf', bound='ClassicalDataStoreReader')


class ClassicalDataStoreReader(abc.ABC):
    @abc.abstractmethod
    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        """Gets the measurement keys in the order they were stored."""

    @abc.abstractmethod
    def get_int(self, key: 'cirq.MeasurementKey') -> int:
        """Gets the integer corresponding to the measurement.

        The integer is determined by summing the qubit-dimensional basis value
        of each measured qubit. For example, if the measurement of qubits
        [q1, q0] produces [1, 0], then the corresponding integer is 2, the big-
        endian equivalent. If they are qutrits and the measurement is [2, 1],
        then the integer is 2 * 3 + 1 = 7.

        Args:
            key: The measurement key.

        Raises:
            KeyError: If the key has not been used.
        """

    @abc.abstractmethod
    def get_digits(self, key: 'cirq.MeasurementKey') -> Tuple[int, ...]:
        """Gets the values of the qubits that were measured into this key.

        For example, if the measurement of qubits [q0, q1] produces [0, 1],
        this function will return (0, 1).

        Args:
            key: The measurement key.

        Raises:
            KeyError: If the key has not been used.
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
        _measurements: Dict['cirq.MeasurementKey', Tuple[int, ...]] = None,
        _measured_qubits: Dict['cirq.MeasurementKey', Tuple['cirq.Qid', ...]] = None,
        _channel_measurements: Dict['cirq.MeasurementKey', int] = None,
        _measurement_types: Dict['cirq.MeasurementKey', 'cirq.MeasurementType'] = None,
    ):
        """Initializes a `ClassicalDataDictionaryStore` object."""
        if not _measurement_types:
            _measurement_types = {}
            if _measurements:
                _measurement_types.update(
                    {k: MeasurementType.MEASUREMENT for k, v in _measurements.items()}
                )
            if _channel_measurements:
                _measurement_types.update(
                    {k: MeasurementType.CHANNEL for k, v in _channel_measurements.items()}
                )
        if _measurements is None:
            _measurements = {}
        if _measured_qubits is None:
            _measured_qubits = {}
        if _channel_measurements is None:
            _channel_measurements = {}
        self._measurements: Dict['cirq.MeasurementKey', Tuple[int, ...]] = _measurements
        self._measured_qubits: Dict[
            'cirq.MeasurementKey', Tuple['cirq.Qid', ...]
        ] = _measured_qubits
        self._channel_measurements: Dict['cirq.MeasurementKey', int] = _channel_measurements
        self._measurement_types: Dict[
            'cirq.MeasurementKey', 'cirq.MeasurementType'
        ] = _measurement_types

    @property
    def measurements(self) -> Mapping['cirq.MeasurementKey', Tuple[int, ...]]:
        """Gets the a mapping from measurement key to measurement."""
        return self._measurements

    @property
    def channel_measurements(self) -> Mapping['cirq.MeasurementKey', int]:
        """Gets the a mapping from measurement key to channel measurement."""
        return self._channel_measurements

    @property
    def measured_qubits(self) -> Mapping['cirq.MeasurementKey', Tuple['cirq.Qid', ...]]:
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
        if key in self._measurement_types:
            raise ValueError(f"Measurement already logged to key {key}")
        self._measurement_types[key] = MeasurementType.MEASUREMENT
        self._measurements[key] = tuple(measurement)
        self._measured_qubits[key] = tuple(qubits)

    def record_channel_measurement(self, key: 'cirq.MeasurementKey', measurement: int):
        if key in self._measurement_types:
            raise ValueError(f"Measurement already logged to key {key}")
        self._measurement_types[key] = MeasurementType.CHANNEL
        self._channel_measurements[key] = measurement

    def get_digits(self, key: 'cirq.MeasurementKey') -> Tuple[int, ...]:
        return (
            self._measurements[key]
            if self._measurement_types[key] == MeasurementType.MEASUREMENT
            else (self._channel_measurements[key],)
        )

    def get_int(self, key: 'cirq.MeasurementKey') -> int:
        if key not in self._measurement_types:
            raise KeyError(f'The measurement key {key} is not in {self._measurements}')
        measurement_type = self._measurement_types[key]
        if measurement_type == MeasurementType.CHANNEL:
            return self._channel_measurements[key]
        if key not in self._measured_qubits:
            return digits.big_endian_bits_to_int(self._measurements[key])
        return digits.big_endian_digits_to_int(
            self._measurements[key], base=[q.dimension for q in self._measured_qubits[key]]
        )

    def copy(self):
        return ClassicalDataDictionaryStore(
            _measurements=self._measurements.copy(),
            _measured_qubits=self._measured_qubits.copy(),
            _channel_measurements=self._channel_measurements.copy(),
            _measurement_types=self._measurement_types.copy(),
        )

    def _json_dict_(self):
        return {
            'measurements': list(self.measurements.items()),
            'measured_qubits': list(self.measured_qubits.items()),
            'channel_measurements': list(self.channel_measurements.items()),
            'measurement_types': list(self.measurement_types.items()),
        }

    @classmethod
    def _from_json_dict_(
        cls, measurements, measured_qubits, channel_measurements, measurement_types, **kwargs
    ):
        return cls(
            _measurements=dict(measurements),
            _measured_qubits=dict(measured_qubits),
            _channel_measurements=dict(channel_measurements),
            _measurement_types=dict(measurement_types),
        )

    def __repr__(self):
        return (
            f'cirq.ClassicalDataDictionaryStore(_measurements={self.measurements!r},'
            f' _measured_qubits={self.measured_qubits!r},'
            f' _channel_measurements={self.channel_measurements!r},'
            f' _measurement_types={self.measurement_types!r})'
        )

    def _value_equality_values_(self):
        return (
            self._measurements,
            self._channel_measurements,
            self._measurement_types,
            self._measured_qubits,
        )
