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


class MapLargeExpandedCircuit:
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
        self.circuit = cirq.Circuit(
            [one_q_x_moment, two_q_cx_moment, one_q_y_moment, two_q_cz_moment] * (num_moments // 4)
        )

    def time_map_moments(self, num_qubits: int, _):
        all_qubits = cirq.LineQubit.range(num_qubits)

        def map_func(m: cirq.Moment, _) -> cirq.Moment:
            new_ops = [op.with_tags("old op") for op in m.operations]
            new_ops += [
                cirq.Z(q).with_tags("new op")
                for q in all_qubits
                if not m.operates_on_single_qubit(q)
            ]
            return cirq.Moment(new_ops)

        _ = cirq.map_moments(circuit=self.circuit, map_func=map_func)

    def time_map_operations_apply_tag(self, *_):
        def map_func(op: cirq.Operation, _) -> cirq.Operation:
            return op.with_tags("old op")

        _ = cirq.map_operations(circuit=self.circuit, map_func=map_func)

    def time_map_operations_to_optree(self, *_):
        def map_func(op: cirq.Operation, _) -> cirq.OP_TREE:
            return [op, op]

        _ = cirq.map_operations(circuit=self.circuit, map_func=map_func)

    def time_map_operations_to_optree_and_unroll(self, *_):
        def map_func(op: cirq.Operation, _) -> cirq.OP_TREE:
            return [op, op]

        _ = cirq.map_operations_and_unroll(circuit=self.circuit, map_func=map_func)
