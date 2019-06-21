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
import asyncio
from typing import Optional, Any, Iterable, Union, List

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
                 tag: Any = None):
        """
        Args:
            circuit: The circuit to sample from.
            repetitions: How many times to sample the circuit.
            tag: An arbitrary value associated with the job. This value is used
                so that when a job completes and is handed back, it is possible
                to tell what the job was for. For example, the key could be a
                string like "main_run" or "calibration_run", or it could be set
                to the component of the Hamiltonian (e.g. a PauliString) that
                the circuit is supposed to be helping to estimate.
        """
        self.circuit = circuit
        self.repetitions = repetitions
        self.tag = tag

    def _value_equality_values_(self):
        return self.circuit, self.repetitions, self.tag

    def __repr__(self):
        return ('cirq.CircuitSampleJob('
                'tag={!r}, repetitions={!r}, circuit={!r})').format(
                    self.tag, self.repetitions, self.circuit)


CIRCUIT_SAMPLE_JOB_TREE = Union[CircuitSampleJob, Iterable[Any]]


class SampleCollector(metaclass=abc.ABCMeta):
    """An interface for concurrently collecting sample data."""

    @abc.abstractmethod
    def next_job(self) -> Optional[CIRCUIT_SAMPLE_JOB_TREE]:
        """Determines what to sample next.

        This method is called by driving code when more samples can be
        requested.

        Returns:
            A CircuitSampleJob describing the circuit to sample, how many
            samples to take, and a key value that can be used in the
            `on_job_result` method to recognize which job this is.

            Can also return a nested iterable of such jobs.

            Returning None or an empty list indicates that there will be no more
            jobs until more results are processed.
        """

    @abc.abstractmethod
    def on_job_result(self, job: CircuitSampleJob,
                      result: study.TrialResult) -> None:
        """Incorporates sampled results.

        This method is called by driving code when sample results have become available.

        The results should be incorporated into the collector's state.
        """

    def collect(self,
                sampler: 'sampler.Sampler',
                *,
                concurrency: int = 2,
                max_total_samples: Optional[int] = None
                ) -> Any:
        """Collects needed samples.

        Examples:

            ```
            collector = cirq.PauliStringCollector(...)
            sampler.collect(collector, concurrency=3)
            print(collector.estimated_energy())
            ```

        See Also:

            Python 3 documentation "Coroutines and Tasks"
            https://docs.python.org/3/library/asyncio-task.html

        Args:
            sampler: The simulator or service to collect samples from.
            concurrency: Desired number of sampling jobs to have in flight at
                any given time.
            max_total_samples: Optional limit on the maximum number of samples
                to collect.

        Returns:
            The collector's result after all desired samples have been
            collected.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.collect_async(sampler,
                               concurrency=concurrency,
                               max_total_samples=max_total_samples))

    async def collect_async(self,
                            sampler: 'sampler.Sampler',
                            *,
                            concurrency: int = 2,
                            max_total_samples: Optional[int] = None
                            ) -> Any:
        """Asynchronously collects needed samples.

        Examples:

            ```
            collector = cirq.PauliStringCollector(...)
            await sampler.collect_async(collector, concurrency=3)
            print(collector.estimated_energy())
            ```

        See Also:

            Python 3 documentation "Coroutines and Tasks"
            https://docs.python.org/3/library/asyncio-task.html

        Args:
            sampler: The simulator or service to collect samples from.
            concurrency: Desired number of sampling jobs to have in flight at
                any given time.
            max_total_samples: Optional limit on the maximum number of samples
                to collect.

        Returns:
            The collector's result after all desired samples have been collected.
        """
        pool = work_pool.CompletionOrderedAsyncWorkPool()
        queued_jobs = []
        remaining_samples = (np.infty
                             if max_total_samples is None
                             else max_total_samples)

        async def _start_async_job(job):
            return job, await sampler.run_async(job.circuit,
                                                repetitions=job.repetitions)

        # Keep dispatching and processing work.
        while True:
            # Fill up the work pool.
            while remaining_samples > 0 and pool.num_uncollected < concurrency:
                if not queued_jobs:
                    queued_jobs.extend(_flatten_jobs(self.next_job()))

                # If no jobs were given, stop asking until something completes.
                if not queued_jobs:
                    break

                # Start new sampling job.
                new_job = queued_jobs.pop(0)
                remaining_samples -= new_job.repetitions
                pool.include_work(_start_async_job(new_job))

            # If no jobs were started or running, we're in a steady state. Halt.
            if not pool.num_uncollected:
                break

            # Forward next job result from pool.
            done_job, done_val = await pool.__anext__()
            self.on_job_result(done_job, done_val)


def _flatten_jobs(given: Optional[CIRCUIT_SAMPLE_JOB_TREE]):
    out = []  # type: List[CircuitSampleJob]
    if given is not None:
        _flatten_jobs_helper(out, given)
    return out


def _flatten_jobs_helper(out: List[CircuitSampleJob], given: CIRCUIT_SAMPLE_JOB_TREE):
    if isinstance(given, CircuitSampleJob):
        out.append(given)
    elif given is not None:
        for item in given:
            _flatten_jobs_helper(out, item)
