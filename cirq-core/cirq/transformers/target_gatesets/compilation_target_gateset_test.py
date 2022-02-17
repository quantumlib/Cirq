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
import numpy as np


class DummyTargetGateset(cirq.TwoQubitAnalyticalCompilationTarget):
    def __init__(self):
        super().__init__(cirq.AnyUnitaryGateFamily(1), cirq.CNOT)

    def _decompose_two_qubit_matrix_to_operations(
        self, q0: cirq.Qid, q1: cirq.Qid, mat: np.ndarray
    ) -> cirq.OP_TREE:
        return [cirq.X.on_each(q0, q1), cirq.CNOT(q0, q1), cirq.Y.on_each(q0, q1)]


def test_two_qubit_analytical_compilation():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(
        cirq.Moment(cirq.Z(q[1])),
        cirq.Moment(cirq.X(q[0])),
        cirq.Moment(cirq.CZ(*q).with_tags("no_compile")),
        cirq.Moment(cirq.Z.on_each(*q)),
        cirq.Moment(cirq.X(q[0])),
        cirq.Moment(cirq.CZ(*q)),
        cirq.Moment(cirq.Z.on_each(*q)),
        cirq.Moment(cirq.X(q[0])),
    )
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
0: ───────X───@['no_compile']───Z───X───@───Z───X───
              │                         │
1: ───Z───────@─────────────────Z───────@───Z───────
''',
    )
    gateset = DummyTargetGateset()
    assert gateset.decompose_to_target_gateset(c_orig[2][q[0]], 2) is NotImplemented
    merged_op = c_orig[1][q[0]].with_tags(gateset._intermediate_result_tag)
    assert gateset.decompose_to_target_gateset(merged_op, 1) is merged_op
    c_new = cirq.convert_to_target_gateset(
        c_orig,
        gateset=DummyTargetGateset(),
        context=cirq.TransformerContext(tags_to_ignore=("no_compile",)),
    )
    cirq.testing.assert_has_diagram(
        c_new,
        '''
0: ───PhXZ(a=-1,x=1,z=0)──────@['no_compile']───X───@───Y───
                              │                     │
1: ───PhXZ(a=-0.5,x=0,z=-1)───@─────────────────X───X───Y───
''',
    )
