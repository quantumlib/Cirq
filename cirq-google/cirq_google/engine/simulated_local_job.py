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
from typing import cast, List, Optional, Sequence, Tuple

import cirq
from cirq_google.engine.client import quantum
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.abstract_local_job import AbstractLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType


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
        self._state = quantum.enums.ExecutionStatus.State.READY
        self._type = simulation_type
        self._failure_code = ''
        self._failure_message = ''

    def execution_status(self) -> quantum.enums.ExecutionStatus.State:
        """Return the execution status of the job."""
        return self._state

    def failure(self) -> Optional[Tuple[str, str]]:
        """Return failure code and message of the job if present."""
        return (self._failure_code, self._failure_message)

    def cancel(self) -> None:
        """Cancel the job."""
        self._state = quantum.enums.ExecutionStatus.State.CANCELLED

    def delete(self) -> None:
        """Deletes the job and result, if any."""
        self.program().delete_job(self.id())
        self._state = quantum.enums.ExecutionStatus.State.STATE_UNSPECIFIED

    def batched_results(self) -> Sequence[Sequence[cirq.Result]]:
        """Returns the job results, blocking until the job is complete.

        This method is intended for batched jobs.  Instead of flattening
        results into a single list, this will return a Sequence[Result]
        for each circuit in the batch.
        """
        if self._type == LocalSimulationType.SYNCHRONOUS:
            reps, sweeps = self.get_repetitions_and_sweeps()
            parent = self.program()
            programs = [parent.get_circuit(n) for n in range(parent.batch_size())]
            try:
                self._state = quantum.enums.ExecutionStatus.State.SUCCESS
                return self._sampler.run_batch(
                    programs=programs,
                    params_list=cast(List[cirq.Sweepable], sweeps),
                    repetitions=reps,
                )
            except Exception as e:
                self._failure_code = '500'
                self._failure_message = str(e)
                self._state = quantum.enums.ExecutionStatus.State.FAILURE
                raise e
        raise ValueError('Unsupported simulation type {self._type}')

    def results(self) -> Sequence[cirq.Result]:
        """Returns the job results, blocking until the job is complete."""
        if self._type == LocalSimulationType.SYNCHRONOUS:
            reps, sweeps = self.get_repetitions_and_sweeps()
            program = self.program().get_circuit()
            try:
                self._state = quantum.enums.ExecutionStatus.State.SUCCESS
                return self._sampler.run_sweep(
                    program=program, params=sweeps[0] if sweeps else None, repetitions=reps
                )
            except Exception as e:
                self._failure_code = '500'
                self._failure_message = str(e)
                self._state = quantum.enums.ExecutionStatus.State.FAILURE
                raise e
        raise ValueError('Unsupported simulation type {self._type}')

    def calibration_results(self) -> Sequence[CalibrationResult]:
        """Returns the results of a run_calibration() call.

        This function will fail if any other type of results were returned.
        """
        raise NotImplementedError
