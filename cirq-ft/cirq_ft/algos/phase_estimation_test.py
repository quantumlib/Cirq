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
import pytest
import numpy as np

import cirq
from cirq_ft.algos import KitaevPhaseEstimation


@pytest.mark.parametrize('theta', [0.234, 0.78, 0.54])
def test_kitaev_phase_estimation(theta):
    U = cirq.Z ** (2 * theta)
    bits_of_precision = 8
    error_bound = 0.1
    precision_registers = cirq.NamedQubit.range(bits_of_precision, prefix='c')
    kitaevPhaseEstimationOp = KitaevPhaseEstimation(bits_of_precision, 1, U, cirq.X).on_registers(
        bits_of_precision_register=precision_registers, eigenvector_register=[cirq.q('ev')]
    )
    cirquit = cirq.Circuit(kitaevPhaseEstimationOp)
    cirquit.append(cirq.measure(*precision_registers, key='m'))
    sim = cirq.Simulator()
    result = sim.run(cirquit, repetitions=10)
    theta_estimates = (
        np.sum(2 ** np.arange(bits_of_precision) * result.measurements['m'], axis=1)
        / 2**bits_of_precision
    )
    assert abs(np.average(theta_estimates) - theta) < error_bound
    print(theta_estimates)
