# Copyright 2024 The Cirq Developers
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
import cirq_google as cg


def test_grid_coupler():
    q1 = cirq.GridQubit(1, 2)
    q2 = cirq.GridQubit(2, 2)
    grid_coupler = cg.GridCoupler(q1, q2)
    assert grid_coupler.qubit1 == q1
    assert grid_coupler.qubit2 == q2

    assert str(grid_coupler) == 'c_q1_2_q2_2'
    assert cirq.circuit_diagram_info(grid_coupler) == cirq.CircuitDiagramInfo(
        wire_symbols=('c_q1_2_q2_2',)
    )
    cirq.testing.assert_equivalent_repr(grid_coupler, global_vals={'cirq_google': cg})


def test_eq():
    eq = cirq.testing.EqualsTester()
    q1 = cirq.GridQubit(1, 2)
    q2 = cirq.GridQubit(2, 2)
    q3 = cirq.GridQubit(2, 1)
    eq.make_equality_group(lambda: cg.GridCoupler(q1, q2), lambda: cg.GridCoupler(q2, q1))
    eq.make_equality_group(lambda: cg.GridCoupler(q2, q3), lambda: cg.GridCoupler(q3, q2))
    eq.make_equality_group(lambda: cg.GridCoupler(q1, q3), lambda: cg.GridCoupler(q3, q1))