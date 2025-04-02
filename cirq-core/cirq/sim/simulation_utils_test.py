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

from cirq import testing
from cirq.sim import simulation_utils


@pytest.mark.parametrize('n,m', [(n, m) for n in range(1, 4) for m in range(1, n + 1)])
def test_state_probabilities_by_indices(n: int, m: int):
    np.random.seed(0)
    state = testing.random_superposition(1 << n)
    d = (state.conj() * state).real
    desired_axes = list(np.random.choice(n, m, replace=False))
    not_wanted = [i for i in range(n) if i not in desired_axes]
    got = simulation_utils.state_probabilities_by_indices(d, desired_axes, (2,) * n)
    want = np.transpose(d.reshape((2,) * n), desired_axes + not_wanted)
    want = np.sum(want.reshape((1 << len(desired_axes), -1)), axis=-1)
    np.testing.assert_allclose(want, got)
