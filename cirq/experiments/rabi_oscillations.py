# Copyright 2018 The Cirq Developers
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
from typing import Any

import numpy as np

from cirq import circuits, devices, google, ops, study, value


def run_rabi_oscillations_on_engine(device: devices.Device,
                                    qubit: ops.QubitId,
                                    repetitions_per_point: int = 1000,
                                    points_per_cycle: int = 100,
                                    cycles: int = 10):
    t = value.Symbol('t')
    circuit = circuits.Circuit.from_ops(
        ops.X(qubit)**t,
        ops.measure(qubit))

    eng = google.engine_from_environment()
    job = eng.run_sweep(
        program=circuit,
        repetitions=repetitions_per_point,
        params=study.Linspace(t,
                              start=0,
                              stop=cycles * 2,
                              length=points_per_cycle * cycles + 1),
        device=device)

    averages = []
    for result in job.results():
        averages.append(result.histogram()[1] / 1000)

    return averages


def probability_of(zeroes, ones, t, period_hypothesis, decay_hypothesis):
    p_one = np.sin(t*np.pi*period_hypothesis)**2 * np.exp(decay_hypothesis*t)
    p_zero = 1 - p_one
    entropy = np.log2(p_zero) * p_zero + np.log(p_one) * p_one
    expected_log_loss = entropy * (ones + zeroes)
    log_loss = np.log2(p_zero) * zeroes + np.log(p_one) * ones
