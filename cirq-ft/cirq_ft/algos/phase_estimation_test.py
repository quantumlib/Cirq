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
from abc import ABC
from functools import cached_property

import pytest
import numpy as np
from attr import frozen

import cirq
from cirq_ft import infra
from cirq_ft.algos import KitaevPhaseEstimation

precision = 8
error_bound = 0.1


@frozen
class KitaevExample(KitaevPhaseEstimation):
    bits: int

    @cached_property
    def eigenvector_register(self) -> infra.Register:
        return infra.Register("eigenvector_register", self.bits)


def test_kitaev_phase_estimation_trivial():
    theta = 0
    U = cirq.I
    eigenvector_register = cirq.q('ev')
    precision_registers = cirq.NamedQubit.range(precision, prefix='c')
    op = KitaevExample(precision, U, 1).on_registers(
        phase_reg=precision_registers, eigenvector_register=[eigenvector_register]
    )
    assert (
        abs(
            simulate_theta_estimate(op, precision_registers, cirq.I.on(eigenvector_register))
            - theta
        )
        < error_bound
    )


@pytest.mark.parametrize('theta', [0.234, 0.78, 0.54])
def test_kitaev_phase_estimation_theta(theta):
    U = cirq.Z ** (2 * theta)
    precision_register = cirq.NamedQubit.range(precision, prefix='c')
    eigenvector_register = cirq.q('ev')
    prepare_eigenvector = cirq.X.on(eigenvector_register)
    op = KitaevExample(precision, U, 1).on_registers(
        phase_reg=precision_register, eigenvector_register=[eigenvector_register]
    )
    assert (
        abs(simulate_theta_estimate(op, precision_register, prepare_eigenvector) - theta)
        < error_bound
    )


def test_kitaev_phase_estimation_multi_qubit_eigenvector():
    theta = 0.5
    U = cirq.CZ
    precision_register = cirq.NamedQubit.range(precision, prefix='c')
    eigenvector_register = cirq.NamedQubit.range(2, prefix='ev')
    prepare_eigenvector = cirq.X.on_each(eigenvector_register)
    op = KitaevExample(precision, U, 2).on_registers(
        phase_reg=precision_register, eigenvector_register=eigenvector_register
    )
    assert (
        abs(simulate_theta_estimate(op, precision_register, prepare_eigenvector) - theta)
        < error_bound
    )


def simulate_theta_estimate(op, measurement_register, eig_prep) -> float:
    cirquit = cirq.Circuit(eig_prep, op)
    cirquit.append(cirq.measure(*measurement_register, key='m'))
    sim = cirq.Simulator()
    result = sim.run(cirquit, repetitions=10)
    theta_estimates = (
        np.sum(2 ** np.arange(precision) * result.measurements['m'], axis=1) / 2**precision
    )
    return np.average(theta_estimates)
