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

from typing import Dict

import collections
import numpy as np
import random

from cirq import circuits, devices, google, ops, study, value
from cirq.experiments import bayes


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
        zeros = result.histogram(key=qubit)[0]
        ones = result.histogram(key=qubit)[1]
        averages.append(ones / (ones + zeros))

    return averages


RabiHypothesis = collections.namedtuple('RabiHypothesis',
                                        ('frequency', 'decay'))


def predicted_p_one(time, rabi_hypothesis):
    cycle = np.sin(time*rabi_hypothesis.frequency)
    decay = np.exp(-time*rabi_hypothesis.decay)
    return (cycle * decay)**2


def bayesian_infer_rabi(time_samples: Dict[float, collections.Counter],
                        prior: Dict[RabiHypothesis, float]
                        ) -> Dict[RabiHypothesis, float]:
    return bayes.infer_posterior(
        predict_log_loss=lambda hyp, exp, out: bayes.binomial_log_loss(
            zeros=out[0],
            ones=out[1],
            p_one=predicted_p_one(time=exp, rabi_hypothesis=hyp)),
        data=time_samples,
        prior=prior)


def uniform_bounded_rabi_prior(min_frequency: float,
                               max_frequency: float,
                               num_frequency: int,
                               min_decay: float,
                               max_decay: float,
                               num_decay: int
                               ) -> Dict[RabiHypothesis, float]:
    frequencies = np.linspace(min_frequency, max_frequency, num_frequency)
    decays = np.linspace(min_decay, max_decay, num_decay)
    p = 1 / (num_frequency * num_decay)
    return {
        RabiHypothesis(frequency, decay): p
        for frequency in frequencies
        for decay in decays
    }


def plot_distribution(distribution: Dict[RabiHypothesis, float]):
    import matplotlib.pyplot as plt

    x = []
    y = []
    z = []
    for (frequency, decay), prob in distribution.items():
        x.append(frequency)
        y.append(decay)
        z.append(prob)

    plt.scatter(x, y, s=np.sqrt(np.array(z)) * 1000, alpha=0.5)
    plt.show()


def generate_rabi_data(time, params: RabiHypothesis, count):
    p_one = predicted_p_one(time, params)
    c = collections.Counter()
    for _ in range(count):
        if random.random() < p_one:
            c[1] += 1
        else:
            c[0] += 1
    return c


def generate_rabi_data_scan(min_time,
                            max_time,
                            num_time,
                            params: RabiHypothesis,
                            count) -> Dict[float, collections.Counter]:
    times = np.linspace(min_time, max_time, num_time)

    return {
        t: generate_rabi_data(t, params, count)
        for t in times
    }


def main():
    prior = uniform_bounded_rabi_prior(
        min_frequency=0,
        max_frequency=10,
        num_frequency=200,
        min_decay=0,
        max_decay=10,
        num_decay=50,
    )
    data = generate_rabi_data_scan(min_time=0, max_time=10, num_time=100,
                                   params=RabiHypothesis(2, 3),
                                   count=100)
    posterior = bayesian_infer_rabi(
        time_samples=data,
        prior=prior
    )
    plot_distribution(posterior)


if __name__ == '__main__':
    main()
