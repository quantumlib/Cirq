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

import abc
from typing import Optional, Any, TypeVar, NamedTuple, Iterable, Union, List

import numpy as np

from cirq import circuits, study, value
from cirq.work import sampler, work_pool


@value.value_equality(unhashable=True)
class CircuitSampleJob:
    """Describes a sampling task."""

    def __init__(self,
                 circuit: circuits.Circuit,
                 *,
                 repetitions: int,
                 id: Any = None):
        """
        Args:
            circuit: The circuit to sample from.
            repetitions: How many times to sample the circuit.
            id: An arbitrary value associated with the job. This value is used
                so that when a job completes and is handed back, it is possible
                to tell what the job was for. For example, the key could be a
                string like "main_run" or "calibration_run", or it could be set
                to the component of the Hamiltonian (e.g. a PauliString) that
                the circuit is supposed to be helping to estimate.
        """
        self.circuit = circuit
        self.repetitions = repetitions
        self.id = id

    def _value_equality_values_(self):
        return self.circuit, self.repetitions, self.id

    def __repr__(self):
        return ('cirq.CircuitSampleJob('
                'id={!r}, repetitions={!r}, circuit={!r})').format(
                    self.id, self.repetitions, self.circuit)


class SampleCollector(metaclass=abc.ABCMeta):
    """An interface for concurrently collecting sample data."""

    @abc.abstractmethod
    def next_job(self
                ) -> Union[None, CircuitSampleJob, Iterable[CircuitSampleJob]]:
        """Called by driving code when more sampling can be started.

        Returns:
            A CircuitSampleJob describing the circuit to sample, how many
            samples to take, and a key value that can be used in the
            `on_job_result` method to recognize which job this is.
        """
        pass

    @abc.abstractmethod
    def on_job_result(self, job: CircuitSampleJob,
                      result: study.TrialResult) -> None:
        """Called by driving code when sample results have become available.

        The results should be incorporated into the collector's state.
        """
        pass


async def async_collect_samples(collector: SampleCollector,
                                sampler: sampler.Sampler,
                                *,
                                concurrency: int = 2,
                                max_total_samples: Optional[int] = None
                               ) -> None:
    """Concurrently collects samples from a simulator or hardware sampler.

    Examples:

        ```
        collector = cirq.PauliStringCollector(...)
        await cirq.async_collect_samples(
            sampler=cirq.Simulator(),
            collector=collector,
            concurrency=3)
        print(collector.estimated_energy())
        ```

    See Also:

        Python 3 documentation "Coroutines and Tasks"
        https://docs.python.org/3/library/asyncio-task.html

    Args:
        collector: Determines which circuits to sample next, and processes the
            resulting samples. This object will be mutated.
        sampler: The simulator or hardware to take samples from.
        concurrency: Desired number of sampling jobs to have in flight at any
            given time.
        max_total_samples: Optional limit on the maximum number of samples to
            collect.

    Returns:
        The collector's result after all desired samples have been collected.
    """
    pool = work_pool.CompletionOrderedAsyncWorkPool()
    remaining_samples = (np.infty
                         if max_total_samples is None else max_total_samples)

    async def _start_async_job(job):
        return job, await sampler.async_sample(job.circuit,
                                               repetitions=job.repetitions)

    # Keep dispatching and processing work.
    while True:
        # Fill up the work pool.
        while (remaining_samples > 0 and pool.num_uncollected < concurrency):
            new_jobs = _flatten_jobs(collector.next_job())

            # If no jobs were given, stop asking until something completes.
            if not new_jobs:
                break

            # Start new sampling jobs.
            for new_job in new_jobs:
                remaining_samples -= new_job.repetitions
                pool.include_work(_start_async_job(new_job))

        # If no jobs were started or running, we're in a steady state. Halt.
        if not pool.num_uncollected:
            break

        # Forward next job result from pool.
        done_job, done_val = await pool.__anext__()
        collector.on_job_result(done_job, done_val)


def _flatten_jobs(
        given: Union[None, CircuitSampleJob, Iterable[CircuitSampleJob]]):
    out = []  # type: List[CircuitSampleJob]
    _flatten_jobs_helper(out, given)
    return out


def _flatten_jobs_helper(
        out: List[CircuitSampleJob],
        given: Union[None, CircuitSampleJob, Iterable[CircuitSampleJob]]):
    if isinstance(given, CircuitSampleJob):
        out.append(given)
    elif given is not None:
        for item in given:
            _flatten_jobs_helper(out, item)
