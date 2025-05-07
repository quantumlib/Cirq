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
"""A helper for jobs that have been created on the Quantum Engine."""
import datetime
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import duet
from google.protobuf import any_pb2

import cirq
from cirq_google.api import v1, v2
from cirq_google.cloud import quantum
from cirq_google.engine import abstract_job, calibration, engine_client
from cirq_google.engine.engine_result import EngineResult

if TYPE_CHECKING:
    import cirq_google.engine.engine as engine_base
    from cirq_google.engine.calibration_result import CalibrationResult
    from cirq_google.engine.engine import engine_processor, engine_program

TERMINAL_STATES = [
    quantum.ExecutionStatus.State.SUCCESS,
    quantum.ExecutionStatus.State.FAILURE,
    quantum.ExecutionStatus.State.CANCELLED,
]


def _flatten(result: Sequence[Sequence[EngineResult]]) -> List[EngineResult]:
    return [res for result_list in result for res in result_list]


class EngineJob(abstract_job.AbstractJob):
    """A job created via the Quantum Engine API.

    This job may be in a variety of states. It may be scheduling, it may be
    executing on a machine, or it may have entered a terminal state
    (either succeeding or failing).

    `EngineJob`s can be iterated over, returning `Result`s. These
    `Result`s can also be accessed by index. Note that this will block
    until the results are returned from the Engine service.

    Attributes:
      project_id: A project_id of the parent Google Cloud Project.
      program_id: Unique ID of the program within the parent project.
      job_id: Unique ID of the job within the parent program.
    """

    def __init__(
        self,
        project_id: str,
        program_id: str,
        job_id: str,
        context: 'engine_base.EngineContext',
        _job: Optional[quantum.QuantumJob] = None,
        job_result_future: Optional[
            duet.AwaitableFuture[Union[quantum.QuantumResult, quantum.QuantumJob]]
        ] = None,
    ) -> None:
        """A job submitted to the engine.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            context: Engine configuration and context to use.
            _job: The optional current job state.
            job_result_future: A future to be completed when the job result is available.
                If set, EngineJob will await this future when a caller asks for the job result. If
                the future is completed with a `QuantumJob`, it is assumed that the job has failed.
        """
        self.project_id = project_id
        self.program_id = program_id
        self.job_id = job_id
        self.context = context
        self._job = _job
        self._results: Optional[Sequence[EngineResult]] = None
        self._calibration_results: Optional[Sequence[CalibrationResult]] = None
        self._batched_results: Optional[Sequence[Sequence[EngineResult]]] = None
        self._job_result_future = job_result_future

    def id(self) -> str:
        """Returns the job id."""
        return self.job_id

    def engine(self) -> 'engine_base.Engine':
        """Returns the parent Engine object."""
        import cirq_google.engine.engine as engine_base

        return engine_base.Engine(self.project_id, context=self.context)

    def program(self) -> 'engine_program.EngineProgram':
        """Returns the parent EngineProgram object."""
        import cirq_google.engine.engine_program as engine_program

        return engine_program.EngineProgram(self.project_id, self.program_id, self.context)

    async def _get_job_async(self, return_run_context: bool = False) -> quantum.QuantumJob:
        return await self.context.client.get_job_async(
            self.project_id, self.program_id, self.job_id, return_run_context
        )

    _get_job = duet.sync(_get_job_async)

    def _inner_job(self) -> quantum.QuantumJob:
        if self._job is None:
            self._job = self._get_job()
        return self._job

    async def _refresh_job_async(self) -> quantum.QuantumJob:
        if self._job is None or self._job.execution_status.state not in TERMINAL_STATES:
            self._job = await self._get_job_async()
        return self._job

    _refresh_job = duet.sync(_refresh_job_async)

    def create_time(self) -> datetime.datetime:
        """Returns when the job was created."""
        return self._inner_job().create_time

    def update_time(self) -> datetime.datetime:
        """Returns when the job was last updated."""
        job = self._refresh_job()
        return job.update_time

    def description(self) -> str:
        """Returns the description of the job."""
        return self._inner_job().description

    def set_description(self, description: str) -> 'EngineJob':
        """Sets the description of the job.

        Params:
            description: The new description for the job.

        Returns:
             This EngineJob.
        """
        self._job = self.context.client.set_job_description(
            self.project_id, self.program_id, self.job_id, description
        )
        return self

    def labels(self) -> Dict[str, str]:
        """Returns the labels of the job."""
        return self._inner_job().labels

    def set_labels(self, labels: Dict[str, str]) -> 'EngineJob':
        """Sets (overwriting) the labels for a previously created quantum job.

        Params:
            labels: The entire set of new job labels.

        Returns:
             This EngineJob.
        """
        self._job = self.context.client.set_job_labels(
            self.project_id, self.program_id, self.job_id, labels
        )
        return self

    def add_labels(self, labels: Dict[str, str]) -> 'EngineJob':
        """Adds new labels to a previously created quantum job.

        Params:
            labels: New labels to add to the existing job labels.

        Returns:
             This EngineJob.
        """
        self._job = self.context.client.add_job_labels(
            self.project_id, self.program_id, self.job_id, labels
        )
        return self

    def remove_labels(self, keys: List[str]) -> 'EngineJob':
        """Removes labels with given keys from the labels of a previously
        created quantum job.

        Params:
            label_keys: Label keys to remove from the existing job labels.

        Returns:
            This EngineJob.
        """
        self._job = self.context.client.remove_job_labels(
            self.project_id, self.program_id, self.job_id, keys
        )
        return self

    def processor_ids(self) -> List[str]:
        """Returns the processor ids provided when the job was created."""
        return [
            engine_client._ids_from_processor_name(p)[1]
            for p in self._inner_job().scheduling_config.processor_selector.processor_names
        ]

    def execution_status(self) -> quantum.ExecutionStatus.State:
        """Return the execution status of the job."""
        return self._refresh_job().execution_status.state

    def status(self) -> str:
        """Return the execution status of the job."""
        return self._refresh_job().execution_status.state.name

    def failure(self) -> Optional[Tuple[str, str]]:
        """Return failure code and message of the job if present."""
        if self._inner_job().execution_status.failure:
            failure = self._inner_job().execution_status.failure
            return (failure.error_code.name, failure.error_message)
        return None

    def get_repetitions_and_sweeps(self) -> Tuple[int, List[cirq.Sweep]]:
        """Returns the repetitions and sweeps for the Quantum Engine job.

        Returns:
            A tuple of the repetition count and list of sweeps.
        """
        if self._job is None or self._job.run_context is None:
            self._job = self._get_job(return_run_context=True)
        return _deserialize_run_context(self._job.run_context)

    def get_processor(self) -> 'Optional[engine_processor.EngineProcessor]':
        """Returns the EngineProcessor for the processor the job is/was run on,
        if available, else None."""
        status = self._inner_job().execution_status
        if not status.processor_name:
            return None
        import cirq_google.engine.engine_processor as engine_processor

        ids = engine_client._ids_from_processor_name(status.processor_name)
        return engine_processor.EngineProcessor(ids[0], ids[1], self.context)

    def get_calibration(self) -> Optional[calibration.Calibration]:
        """Returns the recorded calibration at the time when the job was run, if
        one was captured, else None."""
        status = self._inner_job().execution_status
        if not status.calibration_name:
            return None
        ids = engine_client._ids_from_calibration_name(status.calibration_name)
        response = self.context.client.get_calibration(*ids)
        metrics = v2.metrics_pb2.MetricsSnapshot.FromString(response.data.value)
        return calibration.Calibration(metrics)

    def cancel(self) -> None:
        """Cancel the job."""
        self.context.client.cancel_job(self.project_id, self.program_id, self.job_id)

    def delete(self) -> None:
        """Deletes the job and result, if any."""
        self.context.client.delete_job(self.project_id, self.program_id, self.job_id)

    async def results_async(self) -> Sequence[EngineResult]:
        """Returns the job results, blocking until the job is complete."""
        import cirq_google.engine.engine as engine_base

        if self._results is None:
            result_response = await self._await_result_async()
            result = result_response.result
            result_type = result.type_url[len(engine_base.TYPE_PREFIX) :]
            if (
                result_type == 'cirq.google.api.v1.Result'
                or result_type == 'cirq.api.google.v1.Result'
            ):
                v1_parsed_result = v1.program_pb2.Result.FromString(result.value)
                self._results = self._get_job_results_v1(v1_parsed_result)  # pragma: no cover
            elif (
                result_type == 'cirq.google.api.v2.Result'
                or result_type == 'cirq.api.google.v2.Result'
            ):
                v2_parsed_result = v2.result_pb2.Result.FromString(result.value)
                self._results = self._get_job_results_v2(v2_parsed_result)
            else:
                raise ValueError(f'invalid result proto version: {result_type}')
        return self._results

    async def _await_result_async(self) -> quantum.QuantumResult:
        if self._job_result_future is not None:
            response = await self._job_result_future
            if isinstance(response, quantum.QuantumResult):
                return response
            elif isinstance(response, quantum.QuantumJob):
                self._job = response
                _raise_on_failure(response)
            else:
                raise ValueError(
                    'Internal error: The job response type is not recognized.'
                )  # pragma: no cover

        async with duet.timeout_scope(self.context.timeout):  # type: ignore[arg-type]
            while True:
                job = await self._refresh_job_async()
                if job.execution_status.state in TERMINAL_STATES:
                    break
                await duet.sleep(1)
        _raise_on_failure(job)
        response = await self.context.client.get_job_results_async(
            self.project_id, self.program_id, self.job_id
        )
        return response

    def _get_job_results_v1(self, result: v1.program_pb2.Result) -> Sequence[EngineResult]:
        job_id = self.id()
        job_finished = self.update_time()

        trial_results = []
        for sweep_result in result.sweep_results:
            sweep_repetitions = sweep_result.repetitions
            key_sizes = [(m.key, len(m.qubits)) for m in sweep_result.measurement_keys]
            for result in sweep_result.parameterized_results:
                data = result.measurement_results
                measurements = v1.unpack_results(data, sweep_repetitions, key_sizes)

                trial_results.append(
                    EngineResult(
                        params=cirq.ParamResolver(result.params.assignments),
                        measurements=measurements,
                        job_id=job_id,
                        job_finished_time=job_finished,
                    )
                )
        return trial_results

    def _get_job_results_v2(self, result: v2.result_pb2.Result) -> Sequence[EngineResult]:
        sweep_results = v2.results_from_proto(result)
        job_id = self.id()
        job_finished = self.update_time()

        # Flatten to single list to match to sampler api.
        return [
            EngineResult.from_result(result, job_id=job_id, job_finished_time=job_finished)
            for sweep_result in sweep_results
            for result in sweep_result
        ]

    def __str__(self) -> str:
        return (
            f'EngineJob(project_id=\'{self.project_id}\', '
            f'program_id=\'{self.program_id}\', job_id=\'{self.job_id}\')'
        )


def _deserialize_run_context(run_context: any_pb2.Any) -> Tuple[int, List[cirq.Sweep]]:
    import cirq_google.engine.engine as engine_base

    run_context_type = run_context.type_url[len(engine_base.TYPE_PREFIX) :]
    if (
        run_context_type == 'cirq.google.api.v1.RunContext'
        or run_context_type == 'cirq.api.google.v1.RunContext'
    ):
        raise ValueError('deserializing a v1 RunContext is not supported')
    if (
        run_context_type == 'cirq.google.api.v2.RunContext'
        or run_context_type == 'cirq.api.google.v2.RunContext'
    ):
        v2_run_context = v2.run_context_pb2.RunContext.FromString(run_context.value)
        return v2_run_context.parameter_sweeps[0].repetitions, [
            v2.sweep_from_proto(s.sweep) for s in v2_run_context.parameter_sweeps
        ]
    raise ValueError(f'unsupported run_context type: {run_context_type}')


def _raise_on_failure(job: quantum.QuantumJob) -> None:
    execution_status = job.execution_status
    state = execution_status.state
    name = job.name
    if state != quantum.ExecutionStatus.State.SUCCESS:
        if state == quantum.ExecutionStatus.State.FAILURE:
            processor = execution_status.processor_name or 'UNKNOWN'
            error_code = execution_status.failure.error_code
            error_message = execution_status.failure.error_message
            raise RuntimeError(
                f"Job {name} on processor {processor} failed. {error_code.name}: {error_message}"
            )
        elif state in TERMINAL_STATES:
            raise RuntimeError(f'Job {name} failed in state {state.name}.')
        else:
            raise RuntimeError(
                f'Timed out waiting for results. Job {name} is in state {state.name}'
            )
