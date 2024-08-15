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
from typing import Any, Iterator, List, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol

import duet
import numpy as np

from cirq import study, value

if TYPE_CHECKING:
    import cirq


@value.value_equality(unhashable=True)
class CircuitSampleJob:
    """Describes a sampling task."""

    def __init__(self, circuit: 'cirq.AbstractCircuit', *, repetitions: int, tag: Any = None):
        """Inits CircuitSampleJob.

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

    def _value_equality_values_(self) -> Any:
        return self.circuit, self.repetitions, self.tag

    def __repr__(self) -> str:
        return (
            f'cirq.CircuitSampleJob(tag={self.tag!r}, '
            f'repetitions={self.repetitions!r}, circuit={self.circuit!r})'
        )


class CircuitSampleJobTree(Protocol):
    def __iter__(self) -> Iterator[Union[CircuitSampleJob, 'CircuitSampleJobTree']]:
        pass


CIRCUIT_SAMPLE_JOB_TREE = Union[CircuitSampleJob, CircuitSampleJobTree]


class Collector(metaclass=abc.ABCMeta):
    """Collects data from a sampler, in parallel, towards some purpose.

    Child classes must override the `next_job` and `on_job_result` methods,
    which respectively determine what to sample and how to process the results.
    Utility methods on the base class such as `collect` and `collect_async` can
    then be given a sampler to collect from, and will request samples with some
    specified amount of parallelism.
    """

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

            Returning None, an empty list, or any other result which flattens
            into an empty list of work, indicates that the driving code should
            await more results (and pass them into on_job_results) before
            bothering to ask for more jobs again.
        """

    @abc.abstractmethod
    def on_job_result(self, job: CircuitSampleJob, result: study.Result) -> None:
        """Incorporates sampled results.

        This method is called by driving code when sample results have become
        available.

        The results should be incorporated into the collector's state.
        """

    def collect(
        self,
        sampler: 'cirq.Sampler',
        *,
        concurrency: int = 2,
        max_total_samples: Optional[int] = None,
    ) -> None:
        """Collects needed samples from a sampler.

        Examples:

            ```
            collector = cirq.PauliStringCollector(...)
            sampler.collect(collector, concurrency=3)
            print(collector.estimated_energy())
            ```

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
        return duet.run(
            self.collect_async,
            sampler,
            concurrency=concurrency,
            max_total_samples=max_total_samples,
        )

    async def collect_async(
        self,
        sampler: 'cirq.Sampler',
        *,
        concurrency: int = 2,
        max_total_samples: Optional[int] = None,
    ) -> None:
        """Asynchronously collects needed samples from a sampler.

        Examples:

            ```
            collector = cirq.PauliStringCollector(...)
            await sampler.collect_async(collector, concurrency=3)
            print(collector.estimated_energy())
            ```

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
        results: duet.AsyncCollector[Tuple[CircuitSampleJob, 'cirq.Result']] = duet.AsyncCollector()
        job_error = None
        running_jobs = 0
        queued_jobs: List[CircuitSampleJob] = []
        remaining_samples = np.inf if max_total_samples is None else max_total_samples

        async def run_job(job):
            nonlocal job_error
            try:
                result = await sampler.run_async(job.circuit, repetitions=job.repetitions)
            except Exception as error:
                if not job_error:
                    results.error(error)
                    job_error = error
            else:
                if not job_error:
                    results.add((job, result))

        # Keep dispatching and processing work.
        async with duet.new_scope() as scope:
            while True:
                # Fill up the work pool.
                while remaining_samples > 0 and running_jobs < concurrency:
                    if not queued_jobs:
                        queued_jobs.extend(_flatten_jobs(self.next_job()))

                    # If no jobs were given, stop asking until something completes.
                    if not queued_jobs:
                        break

                    # Start new sampling job.
                    new_job = queued_jobs.pop(0)
                    remaining_samples -= new_job.repetitions
                    running_jobs += 1
                    scope.spawn(run_job, new_job)

                # If no jobs are running, we're in a steady state. Halt.
                if not running_jobs:
                    break

                # Get result from next completed job and call on_job_result.
                job, result = await results.__anext__()
                running_jobs -= 1
                self.on_job_result(job, result)


def _flatten_jobs(tree: Optional[CIRCUIT_SAMPLE_JOB_TREE]) -> Iterator[CircuitSampleJob]:
    if isinstance(tree, CircuitSampleJob):
        yield tree
    elif tree is not None:
        for item in tree:
            yield from _flatten_jobs(item)
