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

import abc
import dataclasses
from typing import Dict, Mapping, Sequence, Tuple, TYPE_CHECKING, FrozenSet, List

import sympy

from cirq._compat import proper_repr
from cirq.protocols import json_serialization, measurement_key_protocol as mkp
from cirq.value import digits, measurement_key

if TYPE_CHECKING:
    import cirq


class ClassicalData:
    """Classical data representing measurements and metadata."""

    def __init__(
        self,
        measurements: Dict['cirq.MeasurementKey', Tuple[int, ...]] = None,
        measured_qubits: Dict['cirq.MeasurementKey', Tuple['cirq.Qid', ...]] = None,
    ):
        # if (measurements is None) != (measured_qubits is None):
        #     raise ValueError(
        #         'measurements and measured_qubits must both either be provided or left default.'
        #     )
        if measurements is None:
            measurements: Dict['cirq.MeasurementKey', Tuple[int, ...]] = {}
        if measured_qubits is None:
            measured_qubits: Dict['cirq.MeasurementKey', Tuple['cirq.Qid', ...]] = {}
        # if set(measurements.keys()) != set(measured_qubits.keys()):
        #     raise ValueError('measurements and measured_qubits must contain same keys.')
        self._measurements = measurements
        self._measured_qubits = measured_qubits

    def keys(self) -> Tuple['cirq.MeasurementKey', ...]:
        return tuple(self._measurements.keys())

    @property
    def measurements(self) -> Mapping['cirq.MeasurementKey', Tuple[int, ...]]:
        return self._measurements

    @property
    def measured_qubits(self) -> Mapping['cirq.MeasurementKey', Tuple['cirq.Qid', ...]]:
        return self._measured_qubits

    def record_measurement(
        self, key: 'cirq.MeasurementKey', measurement: Sequence[int], qubits: Sequence['cirq.Qid']
    ):
        if len(measurement) != len(qubits):
            raise ValueError(f'{len(measurement)} measurements but {len(qubits)} qubits.')
        if key in self._measurements:
            raise ValueError(f"Measurement already logged to key {key!r}")
        self._measurements[key] = tuple(measurement)
        self._measured_qubits[key] = tuple(qubits)

    def copy(self):
        return ClassicalData(self._measurements.copy(), self._measured_qubits.copy())
