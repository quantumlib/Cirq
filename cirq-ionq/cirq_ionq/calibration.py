# Copyright 2020 The Cirq Developers
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

import datetime

from typing import Optional, Set, Tuple

import cirq


class Calibration:
    """An object representing the current calibration state of a QPU."""

    def __init__(self, calibration_dict: dict):
        self._calibration_dict = calibration_dict

    def num_qubits(self) -> int:
        """The number of qubits for the QPU."""
        return int(self._calibration_dict['qubits'])

    def target(self) -> str:
        """The name of the QPU."""
        return self._calibration_dict['target']

    def calibration_time(self, tz: Optional[datetime.tzinfo] = None) -> datetime.datetime:
        """Return a python datetime object for the calibration time.

        Args:
            tz: The timezone for the string. If None, the method uses the platform's local timezone.

        Returns:
            A `datetime` object with the time.
        """
        # Python datetime only like microseconds, not milliseconds, and does not like 'Z'.
        first, second = self._calibration_dict['date'].split('.')
        modified_date = f'{first}.{second[:3]}'
        dt = datetime.datetime.strptime(modified_date, '%Y-%m-%dT%H:%M:%S.%f')
        return dt.replace(tzinfo=datetime.timezone.utc).astimezone(tz=tz)

    def fidelities(self) -> dict:
        """Returns the metrics (fidelities)."""
        return self._calibration_dict['fidelity']

    def timings(self) -> dict:
        """Returns the gate, measurement, and reseting timings."""
        return self._calibration_dict['timing']

    def connectivity(self) -> Set[Tuple[cirq.LineQubit, cirq.LineQubit]]:
        """Returns which qubits and can interact with which.

        Returns:
            A set of the possible qubits that can interact as tuples. This contains both
            ordered pairs. If `(cirq.LineQubit(x), cirq.LineQubit(y))` is in the set, then
            `(cirq.LineQubit(y), cirq.LineQubit(x))` is in the set.
        """
        connections = self._calibration_dict['connectivity']
        to_qubit = lambda x: cirq.LineQubit(int(x))
        return set((to_qubit(x), to_qubit(y)) for x, y in connections).union(
            set((to_qubit(y), to_qubit(x)) for x, y in connections)
        )
