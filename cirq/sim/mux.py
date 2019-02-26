# Copyright 2019 The Cirq Developers
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

"""Sampling/simulation methods that delegate to appropriate simulators."""

from typing import List, Type, Union, Optional

import numpy as np

from cirq import circuits, protocols, study, devices, schedules
from cirq.sim import sparse_simulator, density_matrix_simulator


def run(program: Union[circuits.Circuit, schedules.Schedule],
        *,
        param_resolver: Optional[study.ParamResolver] = None,
        repetitions: int = 1,
        noise: devices.NoiseModel = devices.NO_NOISE,
        dtype: Type[np.number] = np.complex64) -> study.TrialResult:
    """Simulates sampling from the given circuit or schedule.

    Args:
        program: The circuit or schedule to sample from.
        param_resolver: Parameters to run with the program.
        repetitions: The number of samples to take.
        noise: Noise model used to to inject noise while taking samples.
        dtype: The `numpy.dtype` used by the simulation. Typically one of
            `numpy.complex64` or `numpy.complex128`.
            Defaults to fast, i.e. to `numpy.complex64`.
    """

    # State vector simulation is much faster, but only works if no randomness.
    if noise is devices.NO_NOISE and protocols.has_unitary(program):
        return sparse_simulator.Simulator(dtype=dtype).run(
            program=program,
            param_resolver=param_resolver,
            repetitions=repetitions)

    return density_matrix_simulator.DensityMatrixSimulator(
        dtype=dtype,
        noise=noise).run(
            program=program,
            param_resolver=param_resolver,
            repetitions=repetitions)


def run_sweep(program: Union[circuits.Circuit, schedules.Schedule],
              params: study.Sweepable,
              *,
              repetitions: int = 1,
              noise: devices.NoiseModel = devices.NO_NOISE,
              dtype: Type[np.number] = np.complex64
              ) -> List[study.TrialResult]:
    """Runs the supplied Circuit or Schedule, mimicking quantum hardware.

    In contrast to run, this allows for sweeping over different parameter
    values.

    Args:
        program: The circuit or schedule to simulate.
        params: Parameters to run with the program.
        repetitions: The number of repetitions to simulate, per set of
            parameter values.
        noise: Noise model used to to inject noise while taking samples.
        dtype: The `numpy.dtype` used by the simulation. Typically one of
            `numpy.complex64` or `numpy.complex128`.
            Defaults to fast, i.e. to `numpy.complex64`.

    Returns:
        TrialResult list for this run; one for each possible parameter
        resolver.
    """
    circuit = (program if isinstance(program, circuits.Circuit)
               else program.to_circuit())
    param_resolvers = study.to_resolvers(params)

    trial_results = []  # type: List[study.TrialResult]
    for param_resolver in param_resolvers:
        measurements = run(circuit,
                           param_resolver=param_resolver,
                           repetitions=repetitions,
                           noise=noise,
                           dtype=dtype)
        trial_results.append(measurements)
    return trial_results
