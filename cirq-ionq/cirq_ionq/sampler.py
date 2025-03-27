# Copyright 2020 The Cirq Developers
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
"""A `cirq.Sampler` implementation for the IonQ API."""

import itertools
from typing import Optional, Sequence, TYPE_CHECKING

import cirq
from cirq_ionq import results

if TYPE_CHECKING:
    import cirq_ionq


class Sampler(cirq.Sampler):
    """A sampler that works against the IonQ API.

    Users should get a sampler from the `sampler` method on `cirq_ionq.Service`.

    Example of using this sampler:
            >> service = cirq_ionq.Service(...)
            >> a, b, c = cirq.LineQubit.range(3)
            >> sampler = service.sampler()
            >> circuit = cirq.Circuit(cirq.X(a), cirq.measure(a, key='out'))
            >> print(sampler.sample(circuit, repetitions=4))
               out
            0    1
            1    1
            2    1
            3    1
    """

    def __init__(
        self,
        service: 'cirq_ionq.Service',
        target: Optional[str],
        timeout_seconds: Optional[int] = None,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ):
        """Construct the sampler.

        Users should get a sampler from the `sampler` method on `cirq_ionq.Service`.

        Args:
            service: The service used to create this sample.
            target: Where to run the job. Can be 'qpu' or 'simulator'. If this is not specified,
                there must be a default target set on `service`.
            seed: If the target is `simulation` the seed for generating results. If None, this
                will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise
                must be a modulate similar to `np.random`.
            timeout_seconds: Length of time to wait for results. Default is specified in the job.
        """
        self._service = service
        self._target = target
        self._seed = seed
        self._timeout_seconds = timeout_seconds

    def run_sweep(
        self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int = 1
    ) -> Sequence[cirq.Result]:
        """Samples from the given Circuit.

        This allows for sweeping over different parameter values,
        unlike the `run` method.  The `params` argument will provide a
        mapping from `sympy.Symbol`s used within the circuit to a set of
        values.  Unlike the `run` method, which specifies a single
        mapping from symbol to value, this method allows a "sweep" of
        values.  This allows a user to specify execution of a family of
        related circuits efficiently.

        Note that this creates jobs for each of the sweeps in the given sweepable, and then
        blocks until all of the jobs are complete.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Either a list of `cirq_ionq.QPUResult` or a list of `cirq_ionq.SimulatorResult`
            depending on whether the job was running on an actual quantum processor or a simulator.
        """
        resolvers = [r for r in cirq.to_resolvers(params)]
        jobs = [
            self._service.create_job(
                circuit=cirq.resolve_parameters(program, resolver),
                repetitions=repetitions,
                target=self._target,
            )
            for resolver in resolvers
        ]
        if self._timeout_seconds is not None:
            job_results = [job.results(timeout_seconds=self._timeout_seconds) for job in jobs]
        else:
            job_results = [job.results() for job in jobs]
        flattened_job_results = list(itertools.chain.from_iterable(job_results))
        cirq_results = []
        for result, params in zip(flattened_job_results, resolvers):
            if isinstance(result, results.QPUResult):
                cirq_results.append(result.to_cirq_result(params=params))
            elif isinstance(result, results.SimulatorResult):
                cirq_results.append(result.to_cirq_result(params=params, seed=self._seed))
        return cirq_results
