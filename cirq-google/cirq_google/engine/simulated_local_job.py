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
"""An implementation of AbstractJob that uses in-memory constructs
and a provided sampler to execute circuits."""
import datetime
from typing import cast, List, Optional, Sequence, Tuple

import concurrent.futures

import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.abstract_local_job import AbstractLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType
from cirq_google.engine.engine_result import EngineResult


def _flatten_results(batch_results: Sequence[Sequence[EngineResult]]) -> List[EngineResult]:
    return [result for batch in batch_results for result in batch]


def _to_engine_results(
    batch_results: Sequence[Sequence['cirq.Result']],
    *,
    job_id: str,
    job_finished_time: datetime.datetime = None,
) -> List[List[EngineResult]]:
    """Convert cirq.Result from simulators into (simulated) EngineResults."""

    if job_finished_time is None:
        job_finished_time = datetime.datetime.now(tz=datetime.timezone.utc)

    return [
        [
            EngineResult.from_result(result, job_id=job_id, job_finished_time=job_finished_time)
            for result in batch
        ]
        for batch in batch_results
    ]


class SimulatedLocalJob(AbstractLocalJob):
    """A quantum job backed by a (local) sampler.

    This class is designed to execute a local simulator using the
    `AbstractEngine` and `AbstractJob` interface.  This class will
    keep track of the status based on the sampler's results.

    If the simulation type is SYNCHRONOUS, the sampler will be called
    once the appropriate results method is called.  Other methods will
    be added later.

    This does not support calibration requests.
    `
    Attributes:
        sampler: Sampler to call for results.
        simulation_type:  Whether sampler execution should be
            synchronous or asynchronous.
    """

    def __init__(
        self,
        *args,
        sampler: cirq.Sampler = None,
        simulation_type: LocalSimulationType = LocalSimulationType.SYNCHRONOUS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._sampler = sampler or cirq.Simulator()
        self._simulation_type = simulation_type
        self._state = quantum.ExecutionStatus.State.READY
        self._type = simulation_type
        self._failure_code = ''
        self._failure_message = ''
        if self._type == LocalSimulationType.ASYNCHRONOUS:
            # If asynchronous mode, just kick off a new task and move on.
            self._thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                self._future = self._thread.submit(self._execute_results)
            finally:
                # We only expect the one future to run in this thread,
                # So we can call shutdown immediately, which will
                # close the thread pool once the future is complete.
                self._thread.shutdown(wait=False)

    def execution_status(self) -> quantum.ExecutionStatus.State:
        """Return the execution status of the job."""
        # Ignore type errors since mypy doesn't handle proto.Enum (a subclass of enum.IntEnum).
        # See https://github.com/python/mypy/issues/6037.
        return self._state  # type: ignore[return-value]

    def failure(self) -> Optional[Tuple[str, str]]:
        """Return failure code and message of the job if present."""
        return (self._failure_code, self._failure_message)

    def cancel(self) -> None:
        """Cancel the job."""
        self._state = quantum.ExecutionStatus.State.CANCELLED

    def delete(self) -> None:
        """Deletes the job and result, if any."""
        self.program().delete_job(self.id())
        self._state = quantum.ExecutionStatus.State.STATE_UNSPECIFIED

    def batched_results(self) -> Sequence[Sequence[EngineResult]]:
        """Returns the job results, blocking until the job is complete.

        This method is intended for batched jobs.  Instead of flattening
        results into a single list, this will return a Sequence[Result]
        for each circuit in the batch.
        """
        if self._type == LocalSimulationType.SYNCHRONOUS:
            return self._execute_results()
        elif self._type == LocalSimulationType.ASYNCHRONOUS:
            return self._future.result()
        else:
            raise ValueError('Unsupported simulation type {self._type}')

    def _execute_results(self) -> Sequence[Sequence[EngineResult]]:
        """Executes the circuit and sweeps on the sampler.

        For synchronous execution, this is called when the results()
        function is called.  For asynchronous execution, this function
        is run in a thread pool that begins when the object is
        instantiated.

        Returns: a List of results from the sweep's execution.
        """
        reps, sweeps = self.get_repetitions_and_sweeps()
        parent = self.program()
        batch_size = parent.batch_size()
        try:
            self._state = quantum.ExecutionStatus.State.RUNNING
            programs = [parent.get_circuit(n) for n in range(batch_size)]
            batch_results = self._sampler.run_batch(
                programs=programs, params_list=cast(List[cirq.Sweepable], sweeps), repetitions=reps
            )
            batch_engine_results = _to_engine_results(batch_results, job_id=self.id())
            self._state = quantum.ExecutionStatus.State.SUCCESS
            return batch_engine_results
        except Exception as e:
            self._failure_code = '500'
            self._failure_message = str(e)
            self._state = quantum.ExecutionStatus.State.FAILURE
            raise e

    def results(self) -> Sequence[EngineResult]:
        """Returns the job results, blocking until the job is complete."""
        if self._type == LocalSimulationType.SYNCHRONOUS:
            return _flatten_results(self._execute_results())
        elif self._type == LocalSimulationType.ASYNCHRONOUS:
            return _flatten_results(self._future.result())
        else:
            raise ValueError('Unsupported simulation type {self._type}')

    def calibration_results(self) -> Sequence[CalibrationResult]:
        """Returns the results of a run_calibration() call.

        This function will fail if any other type of results were returned.
        """
        raise NotImplementedError
