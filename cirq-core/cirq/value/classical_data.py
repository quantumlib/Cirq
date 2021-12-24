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

import dataclasses
from typing import Dict, Mapping, Sequence, Tuple, TYPE_CHECKING

from cirq.protocols import json_serialization
from cirq.value import digits

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass
class ClassicalData:
    """Classical data representing measurements and metadata."""

    _measurements: Dict['cirq.MeasurementKey', Tuple[int, ...]]
    _measured_qubits: Dict['cirq.MeasurementKey', Tuple['cirq.Qid', ...]]

    def __init__(
        self,
        measurements: Dict['cirq.MeasurementKey', Tuple[int, ...]] = None,
        measured_qubits: Dict['cirq.MeasurementKey', Tuple['cirq.Qid', ...]] = None,
    ):
        """Initializes a `ClassicalData` object.

        Args:
            measurements: The measurements to seed with, if any.
            measured_qubits: The qubits corresponding to the measurements.
        """
        # TODO: Uncomment this after log_of_measurement_results is deprecated and removed
        # if (measurements is None) != (measured_qubits is None):
        #     raise ValueError(
        #         'measurements and measured_qubits must both either be provided or left default.'
        #     )
        if measurements is None:
            measurements = {}
        if measured_qubits is None:
            measured_qubits = {}
        # TODO: Uncomment this after log_of_measurement_results is deprecated and removed
        # if set(measurements.keys()) != set(measured_qubits.keys()):
        #     raise ValueError('measurements and measured_qubits must contain same keys.')
        self._measurements = measurements
        self._measured_qubits = measured_qubits

    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        """Gets the measurement keys in the order they were stored."""
        return tuple(self._measurements.keys())

    @property
    def measurements(self) -> Mapping['cirq.MeasurementKey', Tuple[int, ...]]:
        """Gets the a mapping from measurement key to measurement."""
        return self._measurements

    @property
    def measured_qubits(self) -> Mapping['cirq.MeasurementKey', Tuple['cirq.Qid', ...]]:
        """Gets the a mapping from measurement key to the qubits measured."""
        return self._measured_qubits

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
                measured, or if the measurement key was already used.
        """
        if len(measurement) != len(qubits):
            raise ValueError(f'{len(measurement)} measurements but {len(qubits)} qubits.')
        if key in self._measurements:
            raise ValueError(f"Measurement already logged to key {key!r}")
        self._measurements[key] = tuple(measurement)
        self._measured_qubits[key] = tuple(qubits)

    def record_channel_measurement(
        self, key: 'cirq.MeasurementKey', measurement: int, qubits: Sequence['cirq.Qid']
    ):
        """Records a channel measurement.

        Args:
            key: The measurement key to hold the measurement.
            measurement: The measurement result.
            qubits: The qubits that were measured.

        Raises:
            ValueError: If the measurement key was already used.
        """
        if key in self._measurements:
            raise ValueError(f"Measurement already logged to key {key!r}")
        self._measurements[key] = (measurement,)
        self._measured_qubits[key] = tuple(qubits)

    def get_int(self, key: 'cirq.MeasurementKey') -> int:
        """Gets the integer corresponding to the measurement.

        Args:
            key: The measurement key.

        Raises:
            ValueError: If the key has not been used.
        """
        if key not in self._measurements:
            raise KeyError(f'The measurement key {key} is not in {self._measurements}')
        measurement = self._measurements[key]
        if len(measurement) == 1:
            # Needed to support keyed channels
            return measurement[0]
        return digits.big_endian_digits_to_int(
            measurement, base=[q.dimension for q in self._measured_qubits[key]]
        )

    def copy(self):
        """Creates a copy of the object."""
        return ClassicalData(self._measurements.copy(), self._measured_qubits.copy())

    def _json_dict_(self):
        return json_serialization.obj_to_dict_helper(self, ['measurements', 'measured_qubits'])

    @classmethod
    def _from_json_dict_(cls, measurements, measured_qubits, **kwargs):
        return cls(measurements=measurements, measured_qubits=measured_qubits)

    def __repr__(self):
        return (
            f'cirq.ClassicalData(measurements={self.measurements!r},'
            f' measured_qubits={self.measured_qubits!r})'
        )
