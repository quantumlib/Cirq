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

import cirq


def _human_size(num_bytes: int, mod: int = 0, units=(' bytes', 'KB', 'MB', 'GB', 'TB', 'PB')):
    """Returns a human readable string representation of bytes"""
    return (
        f'{num_bytes}.{mod}{units[0]}'
        if num_bytes < 1024
        else _human_size(num_bytes >> 10, num_bytes % 1024, units[1:])
    )


class SerializeLargeExpandedCircuits:
    param_names = ["num_qubits", "num_moments"]
    params = ([100, 500, 1000], [100, 1000, 4000])
    timeout = 600  # Change timeout to 2 minutes instead of default 60 seconds.

    def setup(self, num_qubits: int, num_moments: int):
        qubits = cirq.LineQubit.range(num_qubits)
        one_q_x_moment = cirq.Moment(cirq.X(q) for q in qubits[::2])
        one_q_y_moment = cirq.Moment(cirq.Y(q) for q in qubits[1::2])
        two_q_cx_moment = cirq.Moment(
            cirq.CNOT(q1, q2) for q1, q2 in zip(qubits[::4], qubits[1::4])
        )
        two_q_cz_moment = cirq.Moment(cirq.CZ(q1, q2) for q1, q2 in zip(qubits[::4], qubits[1::4]))
        measurement_moment = cirq.Moment(cirq.measure_each(*qubits))
        self.circuit = cirq.Circuit(
            [one_q_x_moment, two_q_cx_moment, one_q_y_moment, two_q_cz_moment, measurement_moment]
            * (num_moments // 5)
        )

    def time_json_serialization(self, *_):
        _ = cirq.to_json(self.circuit)

    def time_json_serialization_gzip(self, *_):
        _ = cirq.to_json_gzip(self.circuit)

    def track_json_serialization_gzip_size(self, *_):
        return _human_size(len(cirq.to_json_gzip(self.circuit)))
