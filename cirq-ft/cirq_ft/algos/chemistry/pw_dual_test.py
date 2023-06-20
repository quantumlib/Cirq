# Copyright 2023 The Cirq Developers
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

import numpy as np
import pytest
from cirq_ft.algos.chemistry import PreparePWDual, SelectPWDual, SubPreparePWDual
from cirq_ft.algos.state_preparation_test import construct_gate_helper_and_qubit_order
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.infra.t_complexity_protocol import t_complexity

import cirq


@pytest.mark.parametrize('M', range(3, 10))
def test_select_t_complexity(M):
    N = 2 * M**3
    select = SelectPWDual(M=M, control_val=1)
    assert t_complexity(select).t > 12 * N + 8 * np.ceil(np.log2(N))
    assert t_complexity(select).t < 13 * N + 8 * np.ceil(np.log2(N))


@pytest.mark.parametrize('M', range(3, 10))
def test_prepare_t_complexity(M):
    N = 2 * M**3
    Us, Ts, Vs, Vxs = np.random.normal(size=4 * N // 2).reshape((4, N // 2))
    prep = PreparePWDual(M=M, T=Ts, U=Us, V=Vs, Vx=Vxs)
    assert t_complexity(prep).t > 6 * N + int(np.ceil(np.log2(N)))


def test_subprepare_diagram():
    M = 2
    num_spat_orb = M**3
    Us, Ts, Vs, Vxs = np.random.normal(size=4 * num_spat_orb).reshape((4, num_spat_orb))
    prep = SubPreparePWDual.build_from_coefficients(Ts, Us, Vs, Vxs, probability_epsilon=0.025)
    g, qubit_order, _ = construct_gate_helper_and_qubit_order(prep)
    circuit = cirq.Circuit(cirq.decompose_once(g.operation))
    actual_diagram = circuit.to_text_diagram(qubit_order=qubit_order).lstrip("\n").rstrip()
    cirq.testing.assert_has_diagram(
        circuit,
        '''
theta: ──────────────────────────QROM_5─────────────────────────────────────────
                                 │
UV0: ───────────────UNIFORM(3)───In───────────────────×(y)──────────────────────
                    │            │                    │
UV1: ───────────────target───────In───────────────────×(y)──────────────────────
                                 │                    │
px: ────────────────UNIFORM(2)───In───────────────────┼──────×(y)───────────────
                                 │                    │      │
py: ────────────────UNIFORM(2)───In───────────────────┼──────×(y)───────────────
                                 │                    │      │
pz: ────────────────UNIFORM(2)───In───────────────────┼──────×(y)───────────────
                                 │                    │      │
sigma_mu: ──────────H────────────┼────────In(y)───────┼──────┼──────In(y)───────
                                 │        │           │      │      │
altUV0: ─────────────────────────QROM_0───┼───────────×(x)───┼──────┼───────────
                                 │        │           │      │      │
altUV1: ─────────────────────────QROM_0───┼───────────×(x)───┼──────┼───────────
                                 │        │           │      │      │
altpx: ──────────────────────────QROM_1───┼───────────┼──────×(x)───┼───────────
                                 │        │           │      │      │
altpy: ──────────────────────────QROM_2───┼───────────┼──────×(x)───┼───────────
                                 │        │           │      │      │
altpz: ──────────────────────────QROM_3───┼───────────┼──────×(x)───┼───────────
                                 │        │           │      │      │
keep: ───────────────────────────QROM_4───In(x)───────┼──────┼──────In(x)───────
                                          │           │      │      │
less_than_equal: ─────────────────────────+(x <= y)───@──────@──────+(x <= y)───
        ''',
        qubit_order=qubit_order,
    )


@pytest.mark.parametrize('M', range(3, 10))
def test_subprepare_t_complexity(M):
    num_spat_orb = M**3
    N = 2 * num_spat_orb
    Us, Ts, Vs, Vxs = np.random.normal(size=4 * num_spat_orb).reshape((4, num_spat_orb))
    sub_prep = SubPreparePWDual.build_from_coefficients(Ts, Us, Vs, Vxs, probability_epsilon=1e-8)
    assert t_complexity(sub_prep).t > 6 * N + int(np.ceil(np.log2(N)))


def test_notebook():
    execute_notebook('pw_dual')
