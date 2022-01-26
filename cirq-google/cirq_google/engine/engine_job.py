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
import time

from typing import Dict, Iterator, List, Optional, overload, Sequence, Tuple, TYPE_CHECKING

import cirq
from cirq_google.engine import abstract_job, calibration, engine_client
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.client import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.api import v1, v2

if TYPE_CHECKING:
    import datetime
    import cirq_google.engine.engine as engine_base
    from cirq_google.engine.engine import engine_program
    from cirq_google.engine.engine import engine_processor

TERMINAL_STATES = [
    quantum.enums.ExecutionStatus.State.SUCCESS,
    quantum.enums.ExecutionStatus.State.FAILURE,
    quantum.enums.ExecutionStatus.State.CANCELLED,
]


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
        _job: Optional[quantum.types.QuantumJob] = None,
        result_type: ResultType = ResultType.Program,
    ) -> None:
        """A job submitted to the engine.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            job_id: Unique ID of the job within the parent program.
            context: Engine configuration and context to use.
            _job: The optional current job state.
            result_type: What type of results are expected, such as
               batched results or the result of a focused calibration.
        """
        self.project_id = project_id
        self.program_id = program_id
        self.job_id = job_id
        self.context = context
        self._job = _job
        self._results: Optional[Sequence[cirq.Result]] = None
        self._calibration_results: Optional[Sequence[CalibrationResult]] = None
        self._batched_results: Optional[Sequence[Sequence[cirq.Result]]] = None
        self.result_type = result_type

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

    def _inner_job(self) -> quantum.types.QuantumJob:
        if not self._job:
            self._job = self.context.client.get_job(
                self.project_id, self.program_id, self.job_id, False
            )
        return self._job

    def _refresh_job(self) -> quantum.types.QuantumJob:
        if not self._job or self._job.execution_status.state not in TERMINAL_STATES:
            self._job = self.context.client.get_job(
                self.project_id, self.program_id, self.job_id, False
            )
        return self._job

    def create_time(self) -> 'datetime.datetime':
        """Returns when the job was created."""
        return self._inner_job().create_time.ToDatetime()

    def update_time(self) -> 'datetime.datetime':
        """Returns when the job was last updated."""
        self._job = self.context.client.get_job(
            self.project_id, self.program_id, self.job_id, False
        )
        return self._job.update_time.ToDatetime()

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

    def execution_status(self) -> quantum.enums.ExecutionStatus.State:
        """Return the execution status of the job."""
        return self._refresh_job().execution_status.state

    def status(self) -> str:
        """Return the execution status of the job."""
        return quantum.types.ExecutionStatus.State.Name(self._refresh_job().execution_status.state)

    def failure(self) -> Optional[Tuple[str, str]]:
        """Return failure code and message of the job if present."""
        if self._inner_job().execution_status.HasField('failure'):
            failure = self._inner_job().execution_status.failure
            return (
                quantum.types.ExecutionStatus.Failure.Code.Name(failure.error_code),
                failure.error_message,
            )
        return None

    def get_repetitions_and_sweeps(self) -> Tuple[int, List[cirq.Sweep]]:
        """Returns the repetitions and sweeps for the Quantum Engine job.

        Returns:
            A tuple of the repetition count and list of sweeps.
        """
        if not self._job or not self._job.HasField('run_context'):
            self._job = self.context.client.get_job(
                self.project_id, self.program_id, self.job_id, True
            )

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

    def batched_results(self) -> Sequence[Sequence[cirq.Result]]:
        """Returns the job results, blocking until the job is complete.

        This method is intended for batched jobs.  Instead of flattening
        results into a single list, this will return a Sequence[Result]
        for each circuit in the batch.
        """
        self.results()
        if not self._batched_results:
            raise ValueError('batched_results called for a non-batch result.')
        return self._batched_results

    def _wait_for_result(self):
        job = self._refresh_job()
        total_seconds_waited = 0.0
        timeout = self.context.timeout
        while True:
            if timeout and total_seconds_waited >= timeout:
                break
            if job.execution_status.state in TERMINAL_STATES:
                break
            time.sleep(0.5)
            total_seconds_waited += 0.5
            job = self._refresh_job()
        _raise_on_failure(job)
        response = self.context.client.get_job_results(
            self.project_id, self.program_id, self.job_id
        )
        return response.result

    def results(self) -> Sequence[cirq.Result]:
        """Returns the job results, blocking until the job is complete."""
        import cirq_google.engine.engine as engine_base

        if not self._results:
            result = self._wait_for_result()
            result_type = result.type_url[len(engine_base.TYPE_PREFIX) :]
            if (
                result_type == 'cirq.google.api.v1.Result'
                or result_type == 'cirq.api.google.v1.Result'
            ):
                v1_parsed_result = v1.program_pb2.Result.FromString(result.value)
                self._results = _get_job_results_v1(v1_parsed_result)
            elif (
                result_type == 'cirq.google.api.v2.Result'
                or result_type == 'cirq.api.google.v2.Result'
            ):
                v2_parsed_result = v2.result_pb2.Result.FromString(result.value)
                self._results = _get_job_results_v2(v2_parsed_result)
            elif result.Is(v2.batch_pb2.BatchResult.DESCRIPTOR):
                v2_parsed_result = v2.batch_pb2.BatchResult.FromString(result.value)
                self._batched_results = self._get_batch_results_v2(v2_parsed_result)
                self._results = self._flatten(self._batched_results)
            else:
                raise ValueError(f'invalid result proto version: {result_type}')
        return self._results

    def calibration_results(self) -> Sequence[CalibrationResult]:
        """Returns the results of a run_calibration() call.

        This function will fail if any other type of results were returned
        by the Engine.
        """
        import cirq_google.engine.engine as engine_base

        if not self._calibration_results:
            result = self._wait_for_result()
            result_type = result.type_url[len(engine_base.TYPE_PREFIX) :]
            if result_type != 'cirq.google.api.v2.FocusedCalibrationResult':
                raise ValueError(f'Did not find calibration results, instead found: {result_type}')
            parsed_val = v2.calibration_pb2.FocusedCalibrationResult.FromString(result.value)
            cal_results = []
            for layer in parsed_val.results:
                metrics = calibration.Calibration(layer.metrics)
                message = layer.error_message or None
                token = layer.token or None
                ts: Optional[datetime.datetime] = None
                if layer.valid_until_ms > 0:
                    ts = datetime.datetime.fromtimestamp(layer.valid_until_ms / 1000)
                cal_results.append(CalibrationResult(layer.code, message, token, ts, metrics))
            self._calibration_results = cal_results
        return self._calibration_results

    @classmethod
    def _get_batch_results_v2(
        cls, results: v2.batch_pb2.BatchResult
    ) -> Sequence[Sequence[cirq.Result]]:
        trial_results = []
        for result in results.results:
            # Add a new list for the result
            trial_results.append(_get_job_results_v2(result))
        return trial_results

    @classmethod
    def _flatten(cls, result) -> Sequence[cirq.Result]:
        return [res for result_list in result for res in result_list]

    def __iter__(self) -> Iterator[cirq.Result]:
        return iter(self.results())

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, item: int) -> cirq.Result:
        pass

    @overload
    def __getitem__(self, item: slice) -> Sequence[cirq.Result]:
        pass

    def __getitem__(self, item):
        return self.results()[item]

    # pylint: enable=function-redefined

    def __len__(self) -> int:
        return len(self.results())

    def __str__(self) -> str:
        return (
            f'EngineJob(project_id=\'{self.project_id}\', '
            f'program_id=\'{self.program_id}\', job_id=\'{self.job_id}\')'
        )


def _deserialize_run_context(
    run_context: quantum.types.any_pb2.Any,
) -> Tuple[int, List[cirq.Sweep]]:
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


def _get_job_results_v1(result: v1.program_pb2.Result) -> Sequence[cirq.Result]:
    trial_results = []
    for sweep_result in result.sweep_results:
        sweep_repetitions = sweep_result.repetitions
        key_sizes = [(m.key, len(m.qubits)) for m in sweep_result.measurement_keys]
        for result in sweep_result.parameterized_results:
            data = result.measurement_results
            measurements = v1.unpack_results(data, sweep_repetitions, key_sizes)

            trial_results.append(
                cirq.ResultDict(
                    params=cirq.ParamResolver(result.params.assignments),
                    measurements=measurements,
                )
            )
    return trial_results


def _get_job_results_v2(result: v2.result_pb2.Result) -> Sequence[cirq.Result]:
    sweep_results = v2.results_from_proto(result)
    # Flatten to single list to match to sampler api.
    return [trial_result for sweep_result in sweep_results for trial_result in sweep_result]


def _raise_on_failure(job: quantum.types.QuantumJob) -> None:
    execution_status = job.execution_status
    state = execution_status.state
    name = job.name
    if state != quantum.enums.ExecutionStatus.State.SUCCESS:
        if state == quantum.enums.ExecutionStatus.State.FAILURE:
            processor = execution_status.processor_name or 'UNKNOWN'
            error_code = execution_status.failure.error_code
            error_message = execution_status.failure.error_message
            raise RuntimeError(
                "Job {} on processor {} failed. {}: {}".format(
                    name,
                    processor,
                    quantum.types.ExecutionStatus.Failure.Code.Name(error_code),
                    error_message,
                )
            )
        elif state in TERMINAL_STATES:
            raise RuntimeError(
                'Job {} failed in state {}.'.format(
                    name,
                    quantum.types.ExecutionStatus.State.Name(state),
                )
            )
        else:
            raise RuntimeError(
                'Timed out waiting for results. Job {} is in state {}'.format(
                    name, quantum.types.ExecutionStatus.State.Name(state)
                )
            )
