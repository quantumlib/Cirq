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

import cirq_ft
import pytest
from cirq_ft.infra.jupyter_tools import execute_notebook


@pytest.mark.parametrize('dim', [*range(2, 10)])
def test_select_t_complexity(dim):
    select = cirq_ft.SelectHubbard(x_dim=dim, y_dim=dim, control_val=1)
    cost = cirq_ft.t_complexity(select)
    N = 2 * dim * dim
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t == 10 * N + 14 * logN - 8
    assert cost.rotations == 0


@pytest.mark.parametrize('dim', [*range(2, 10)])
def test_prepare_t_complexity(dim):
    prepare = cirq_ft.PrepareHubbard(x_dim=dim, y_dim=dim, t=2, mu=8)
    cost = cirq_ft.t_complexity(prepare)
    logN = 2 * (dim - 1).bit_length() + 1
    assert cost.t <= 32 * logN
    # TODO(#233): The rotation count should reduce to a constant once cost for Controlled-H
    # gates is recognized as $2$ T-gates instead of $2$ rotations.
    assert cost.rotations <= 2 * logN + 9


def test_notebook():
    execute_notebook('hubbard_model')
