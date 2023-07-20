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

precision = 8
error_bound = 0.1


def test_kitaev_phase_estimation_trivial():
    theta = 0
    U = cirq.I
    precision_registers = cirq.NamedQubit.range(precision, prefix='c')
    op = KitaevPhaseEstimation(precision, U, cirq.X).on_registers(
        bits_of_precision_register=precision_registers, eigenvector_register=[cirq.q('ev')]
    )
    assert abs(simulate_theta_estimate(op, precision_registers) - theta) < error_bound


@pytest.mark.parametrize('theta', [0.234, 0.78, 0.54])
def test_kitaev_phase_estimation_theta(theta):
    U = cirq.Z ** (2 * theta)
    precision_register = cirq.NamedQubit.range(precision, prefix='c')
    op = KitaevPhaseEstimation(precision, U, cirq.X).on_registers(
        bits_of_precision_register=precision_register, eigenvector_register=[cirq.q('ev')]
    )
    assert abs(simulate_theta_estimate(op, precision_register) - theta) < error_bound


def simulate_theta_estimate(op, measurement_register) -> float:
    cirquit = cirq.Circuit(op)
    cirquit.append(cirq.measure(*measurement_register, key='m'))
    sim = cirq.Simulator()
    result = sim.run(cirquit, repetitions=10)
    theta_estimates = (
        np.sum(2 ** np.arange(precision) * result.measurements['m'], axis=1) / 2**precision
    )
    return np.average(theta_estimates)
