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

import time

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from cirq import study
from cirq.google.engine import calibration
from cirq.google.engine.client import quantum
from cirq.google.api import v1, v2
import cirq.google.engine.engine as engine_base

if TYPE_CHECKING:
    from cirq.google.engine.engine import engine_program

TERMINAL_STATES = [
    quantum.enums.ExecutionStatus.State.SUCCESS,
    quantum.enums.ExecutionStatus.State.FAILURE,
    quantum.enums.ExecutionStatus.State.CANCELLED
]


class EngineJob:
    """A job created via the Quantum Engine API.

    This job may be in a variety of states. It may be scheduling, it may be
    executing on a machine, or it may have entered a terminal state
    (either succeeding or failing).

    Attributes:
      project_id: A project_id of the parent Google Cloud Project.
      program_id: Unique ID of the program within the parent project.
      job_id: Unique ID of the job within the parent program.
    """

    def __init__(self, job_id: str, program: 'engine_program.EngineProgram',
                 job: quantum.types.QuantumJob) -> None:
        """A job submitted to the engine.

        Args:
            job_id: Unique ID of the job within the parent program.
            program: Parent EngineProgram for the job.
            job: A job proto.
        """
        self.project_id = program.project_id
        self.program_id = program.program_id
        self.job_id = job_id
        self._program = program
        self._job = job
        self._results: Optional[List[study.TrialResult]] = None

    def engine(self) -> 'engine_base.Engine':
        """Returns the parent Engine object.

        Returns:
            The job's parent Engine.
        """
        return self._program.engine()

    def program(self) -> 'engine_program.EngineProgram':
        """Returns the parent EngineProgram object.

        Returns:
            The job's parent EngineProgram.
        """
        return self._program

    def _refresh_job(self) -> 'quantum.types.QuantumJob':
        if self._job.execution_status.state not in TERMINAL_STATES:
            self._job = self.engine().client.get_job(self.project_id,
                                                     self.program_id,
                                                     self.job_id, False)
        return self._job

    def description(self) -> str:
        """Returns the description of the job.

        Returns:
             The current description of the job.
        """
        return self._job.description

    def set_description(self, description: str) -> 'EngineJob':
        """Sets the description of the job.

        Params:
            description: The new description for the job.

        Returns:
             This EngineJob.
        """
        self._job = self.engine().client.set_job_description(
            self.project_id, self.program_id, self.job_id, description)
        return self

    def labels(self) -> Dict[str, str]:
        """Returns the labels of the job.

        Returns:
             The current labels of the job.
        """
        return self._job.labels

    def set_labels(self, labels: Dict[str, str]) -> 'EngineJob':
        """Sets (overwriting) the labels for a previously created quantum job.

        Params:
            labels: The entire set of new job labels.

        Returns:
             This EngineJob.
        """
        self._job = self.engine().client.set_job_labels(self.project_id,
                                                        self.program_id,
                                                        self.job_id, labels)
        return self

    def add_labels(self, labels: Dict[str, str]) -> 'EngineJob':
        """Adds new labels to a previously created quantum job.

        Params:
            labels: New labels to add to the existing job labels.

        Returns:
             This EngineJob.
        """
        self._job = self.engine().client.add_job_labels(self.project_id,
                                                        self.program_id,
                                                        self.job_id, labels)
        return self

    def remove_labels(self, keys: List[str]) -> 'EngineJob':
        """Removes labels with given keys from the labels of a previously
        created quantum job.

        Params:
            label_keys: Label keys to remove from the existing job labels.

        Returns:
            This EngineJob.
        """
        self._job = self.engine().client.remove_job_labels(
            self.project_id, self.program_id, self.job_id, keys)
        return self

    def status(self) -> str:
        """Return the execution status of the job."""
        return quantum.types.ExecutionStatus.State.Name(
            self._refresh_job().execution_status.state)

    def get_repetitions_and_sweeps(self) -> Tuple[int, List[study.Sweep]]:
        """Returns the repetitions and sweeps for the Quantum Engine job.

        Returns:
            A tuple of the repetition count and list of sweeps.
        """
        if not self._job.HasField('run_context'):
            self._job = self.engine().client.get_job(self.project_id,
                                                     self.program_id,
                                                     self.job_id, True)

        return self._deserialize_run_context(self._job.run_context)

    def _deserialize_run_context(self, run_context: quantum.types.any_pb2.Any
                                ) -> Tuple[int, List[study.Sweep]]:
        run_context_type = run_context.type_url[len(engine_base.TYPE_PREFIX):]
        if (run_context_type == 'cirq.google.api.v1.RunContext' or
                run_context_type == 'cirq.api.google.v1.RunContext'):
            raise ValueError('deserializing a v1 RunContext is not supported')
        if (run_context_type == 'cirq.google.api.v2.RunContext' or
                run_context_type == 'cirq.api.google.v2.RunContext'):
            v2_run_context = v2.run_context_pb2.RunContext()
            v2_run_context.ParseFromString(run_context.value)
            return v2_run_context.parameter_sweeps[0].repetitions, [
                v2.sweep_from_proto(s.sweep)
                for s in v2_run_context.parameter_sweeps
            ]
        raise ValueError(
            'unsupported run_context type: {}'.format(run_context_type))

    def get_calibration(self) -> Optional[calibration.Calibration]:
        """Returns the recorded calibration at the time when the job was run, if
        one was captured, else None."""
        status = self._job.execution_status
        if not status.calibration_name:
            return None
        ids = self.engine().client._ids_from_calibration_name(
            status.calibration_name)
        response = self.engine().client.get_calibration(*ids)
        metrics = v2.metrics_pb2.MetricsSnapshot()
        metrics.ParseFromString(response.data.value)
        return calibration.Calibration(metrics)

    def cancel(self) -> None:
        """Cancel the job."""
        self.engine().client.cancel_job(self.project_id, self.program_id,
                                        self.job_id)

    def delete(self) -> None:
        """Deletes the job and result, if any."""
        self.engine().client.delete_job(self.project_id, self.program_id,
                                        self.job_id)

    def results(self) -> List[study.TrialResult]:
        """Returns the job results, blocking until the job is complete.
        """
        if not self._results:
            job = self._refresh_job()
            for _ in range(1000):
                if job.execution_status.state in TERMINAL_STATES:
                    break
                time.sleep(0.5)
                job = self._refresh_job()
            self._raise_on_failure(job)
            response = self.engine().client.get_job_results(
                self.project_id, self.program_id, self.job_id)
            result = response.result
            result_type = result.type_url[len(engine_base.TYPE_PREFIX):]
            if (result_type == 'cirq.google.api.v1.Result' or
                    result_type == 'cirq.api.google.v1.Result'):
                v1_parsed_result = v1.program_pb2.Result()
                v1_parsed_result.ParseFromString(result.value)
                self._results = self._get_job_results_v1(v1_parsed_result)
            elif (result_type == 'cirq.google.api.v2.Result' or
                  result_type == 'cirq.api.google.v2.Result'):
                v2_parsed_result = v2.result_pb2.Result()
                v2_parsed_result.ParseFromString(result.value)
                self._results = self._get_job_results_v2(v2_parsed_result)
            else:
                raise ValueError(
                    'invalid result proto version: {}'.format(result_type))
        return self._results

    def _get_job_results_v1(self, result: v1.program_pb2.Result
                           ) -> List[study.TrialResult]:
        trial_results = []
        for sweep_result in result.sweep_results:
            sweep_repetitions = sweep_result.repetitions
            key_sizes = [
                (m.key, len(m.qubits)) for m in sweep_result.measurement_keys
            ]
            for result in sweep_result.parameterized_results:
                data = result.measurement_results
                measurements = v1.unpack_results(data, sweep_repetitions,
                                                 key_sizes)

                trial_results.append(
                    study.TrialResult.from_single_parameter_set(
                        params=study.ParamResolver(result.params.assignments),
                        measurements=measurements))
        return trial_results

    def _get_job_results_v2(self, result: v2.result_pb2.Result
                           ) -> List[study.TrialResult]:
        sweep_results = v2.results_from_proto(result)
        # Flatten to single list to match to sampler api.
        return [
            trial_result for sweep_result in sweep_results
            for trial_result in sweep_result
        ]

    def _raise_on_failure(self, job: quantum.types.QuantumJob) -> None:
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
                        name, processor,
                        quantum.types.ExecutionStatus.Failure.Code.Name(
                            error_code), error_message))
            elif state in TERMINAL_STATES:
                raise RuntimeError('Job {} failed in state {}.'.format(
                    name,
                    quantum.types.ExecutionStatus.State.Name(state),
                ))
            else:
                raise RuntimeError(
                    'Timed out waiting for results. Job {} is in state {}'.
                    format(name,
                           quantum.types.ExecutionStatus.State.Name(state)))

    def __iter__(self):
        return iter(self.results())

    def __str__(self):
        return str('EngineJob({}, {}, {})'.format(self.project_id,
                                                  self.program_id, self.job_id))
