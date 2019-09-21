# Copyright 2018 The Cirq Developers
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
"""An optimization pass to put as many measurements possible at the end."""

from cirq import circuits, ops, protocols


class SynchronizeTerminalMeasurements():
    """Move measurements to the end.

    Move all terminal measurements in a circuit to the final moment if it
    can accomodate them (without overlapping with other operations). If
    self._measure_only_moment is true then a new moment will be added to
    the end of the circuit containing all the measurements that could be
    brough forward.
    """

    def __init__(self, measurements_only_moment: bool = True):
        """
        Args:
            measurements_only_moment: Bool indicating whether or not all
                measurements should be moved to the last existing moment so
                that the circuit might contain a mixture of measurements and
                other ops (False), or a new moment containing them should be
                added to the end with the guarantee to contain only
                measurements (True).
        """
        self._measurement_only_moment = measurements_only_moment

    def __call__(self, circuit: circuits.Circuit):
        self.optimize_circuit(circuit)

    def optimize_circuit(self, circuit: circuits.Circuit) -> None:
        deletions = []
        terminal_measures = set()
        deepest_measurement = -1
        all_measurements = circuit.findall_operations(protocols.is_measurement)
        for index, op in all_measurements:
            if circuit.next_moment_operating_on(op.qubits, index + 1) is None:
                deepest_measurement = max(deepest_measurement, index)
                deletions.append((index, op))
                terminal_measures.add(op)

        circuit.batch_remove(deletions)
        if not self._measurement_only_moment or circuit[
            deepest_measurement] == ops.Moment([]):
            # Can safely add to deepest measurement if
            # self._measurement_only_moment is false or we happen to get an
            # empty final moment before re-adding all the measurements.
            for op in terminal_measures:
                circuit[deepest_measurement] = circuit[
                    deepest_measurement].with_operation(op)
        else:
            circuit.append(ops.Moment(list(terminal_measures)))
