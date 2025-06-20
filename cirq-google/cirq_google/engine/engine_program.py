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

import datetime
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union

import duet
from google.protobuf import any_pb2

import cirq
from cirq_google.engine import abstract_program, engine_client
from cirq_google.cloud import quantum
from cirq_google.api import v2
from cirq_google.engine import engine_job
from cirq_google.serialization import circuit_serializer

if TYPE_CHECKING:
    import cirq_google.engine.engine as engine_base


class EngineProgram(abstract_program.AbstractProgram):
    """A program created via the Quantum Engine API.

    This program wraps a Circuit with additional metadata used to
    schedule against the devices managed by Quantum Engine.

    Attributes:
        project_id: A project_id of the parent Google Cloud Project.
        program_id: Unique ID of the program within the parent project.
    """

    def __init__(
        self,
        project_id: str,
        program_id: str,
        context: 'engine_base.EngineContext',
        _program: Optional[quantum.QuantumProgram] = None,
    ) -> None:
        """A job submitted to the engine.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            context: Engine configuration and context to use.
            _program: The optional current program state.
        """
        self.project_id = project_id
        self.program_id = program_id
        self.context = context
        self._program = _program

    async def run_sweep_async(
        self,
        processor_id: str,
        job_id: Optional[str] = None,
        params: cirq.Sweepable = None,
        repetitions: int = 1,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        *,
        run_name: str = "",
        device_config_name: str = "",
    ) -> engine_job.EngineJob:
        """Runs the program on the QuantumEngine.

        In contrast to run, this runs across multiple parameter sweeps, and
        does not block until a result is returned.

        Args:
            job_id: Optional job id to use. If this is not provided, a random id
                of the format 'job-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            params: Parameters to run with the program.
            repetitions: The number of circuit repetitions to run.
            description: An optional description to set on the job.
            labels: Optional set of labels to set on the job.
            processor_id: Processor id for running the program.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.

        Raises:
            ValueError: If a processor id hasn't been specified to run the job
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
            ValueError: If either `run_name` and `device_config_name` are set but
                `processor_id` is empty.
        """
        import cirq_google.engine.engine as engine_base

        if not job_id:
            job_id = engine_base._make_random_id('job-')
        run_context = self.context._serialize_run_context(params, repetitions)

        created_job_id, job = await self.context.client.create_job_async(
            project_id=self.project_id,
            program_id=self.program_id,
            job_id=job_id,
            processor_id=processor_id,
            run_context=run_context,
            description=description,
            labels=labels,
            run_name=run_name,
            device_config_name=device_config_name,
        )
        return engine_job.EngineJob(
            self.project_id, self.program_id, created_job_id, self.context, job
        )

    run_sweep = duet.sync(run_sweep_async)

    async def run_async(
        self,
        processor_id: str,
        job_id: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
        repetitions: int = 1,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        *,
        run_name: str = "",
        device_config_name: str = "",
    ) -> cirq.Result:
        """Runs the supplied Circuit via Quantum Engine.

        Args:
            job_id: Optional job id to use. If this is not provided, a random id
                of the format 'job-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            description: An optional description to set on the job.
            labels: Optional set of labels to set on the job.
            processor_id: Processor id for running the program.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.

        Returns:
            A single Result for this run.

        Raises:
            ValueError: If a processor id hasn't been specified to run the job
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
            ValueError: If either `run_name` and `device_config_name` are set but
                `processor_id` is empty.
        """
        job = await self.run_sweep_async(
            job_id=job_id,
            params=[param_resolver],
            repetitions=repetitions,
            processor_id=processor_id,
            description=description,
            labels=labels,
            run_name=run_name,
            device_config_name=device_config_name,
        )
        results = await job.results_async()
        return results[0]

    run = duet.sync(run_async)

    def engine(self) -> 'engine_base.Engine':
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.
        """
        import cirq_google.engine.engine as engine_base

        return engine_base.Engine(self.project_id, context=self.context)

    def get_job(self, job_id: str) -> engine_job.EngineJob:
        """Returns an EngineJob for an existing Quantum Engine job.

        Args:
            job_id: Unique ID of the job within the parent program.

        Returns:
            A EngineJob for the job.
        """
        return engine_job.EngineJob(self.project_id, self.program_id, job_id, self.context)

    async def list_jobs_async(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
        execution_states: Optional[Set[quantum.ExecutionStatus.State]] = None,
    ) -> Sequence[engine_job.EngineJob]:
        """Returns the list of jobs for this program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            created_after: retrieve jobs that were created after this date
                or time.
            created_before: retrieve jobs that were created after this date
                or time.
            has_labels: retrieve jobs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}

            execution_states: retrieve jobs that have an execution state  that
                is contained in `execution_states`. See
                `quantum.ExecutionStatus.State` enum for accepted values.
        """
        client = self.context.client
        response = await client.list_jobs_async(
            self.project_id,
            self.program_id,
            created_before=created_before,
            created_after=created_after,
            has_labels=has_labels,
            execution_states=execution_states,
        )
        return [
            engine_job.EngineJob(
                project_id=engine_client._ids_from_job_name(j.name)[0],
                program_id=engine_client._ids_from_job_name(j.name)[1],
                job_id=engine_client._ids_from_job_name(j.name)[2],
                context=self.context,
                _job=j,
            )
            for j in response
        ]

    list_jobs = duet.sync(list_jobs_async)

    def _inner_program(self) -> quantum.QuantumProgram:
        if self._program is None:
            self._program = self.context.client.get_program(self.project_id, self.program_id, False)
        return self._program

    def create_time(self) -> 'datetime.datetime':
        """Returns when the program was created."""
        return self._inner_program().create_time

    def update_time(self) -> 'datetime.datetime':
        """Returns when the program was last updated."""
        self._program = self.context.client.get_program(self.project_id, self.program_id, False)
        return self._program.update_time

    def description(self) -> str:
        """Returns the description of the program."""
        return self._inner_program().description

    async def set_description_async(self, description: str) -> 'EngineProgram':
        """Sets the description of the program.

        Params:
            description: The new description for the program.

        Returns:
             This EngineProgram.
        """
        self._program = await self.context.client.set_program_description_async(
            self.project_id, self.program_id, description
        )
        return self

    set_description = duet.sync(set_description_async)

    def labels(self) -> Dict[str, str]:
        """Returns the labels of the program."""
        return self._inner_program().labels

    async def set_labels_async(self, labels: Dict[str, str]) -> 'EngineProgram':
        """Sets (overwriting) the labels for a previously created quantum
        program.

        Params:
            labels: The entire set of new program labels.

        Returns:
             This EngineProgram.
        """
        self._program = await self.context.client.set_program_labels_async(
            self.project_id, self.program_id, labels
        )
        return self

    set_labels = duet.sync(set_labels_async)

    async def add_labels_async(self, labels: Dict[str, str]) -> 'EngineProgram':
        """Adds new labels to a previously created quantum program.

        Params:
            labels: New labels to add to the existing program labels.

        Returns:
             This EngineProgram.
        """
        self._program = await self.context.client.add_program_labels_async(
            self.project_id, self.program_id, labels
        )
        return self

    add_labels = duet.sync(add_labels_async)

    async def remove_labels_async(self, keys: List[str]) -> 'EngineProgram':
        """Removes labels with given keys from the labels of a previously
        created quantum program.

        Params:
            label_keys: Label keys to remove from the existing program labels.

        Returns:
             This EngineProgram.
        """
        self._program = await self.context.client.remove_program_labels_async(
            self.project_id, self.program_id, keys
        )
        return self

    remove_labels = duet.sync(remove_labels_async)

    async def get_circuit_async(self) -> cirq.Circuit:
        """Returns the cirq Circuit for the Quantum Engine program. This is only
        supported if the program was created with the V2 protos.

        Returns:
            The program's cirq Circuit.
        """
        if self._program is None or self._program.code is None:
            self._program = await self.context.client.get_program_async(
                self.project_id, self.program_id, True
            )
        return _deserialize_program(self._program.code)

    get_circuit = duet.sync(get_circuit_async)

    async def batch_size_async(self) -> int:
        """Returns the number of programs in a batch program.

        Raises:
            ValueError: if the program created was not a batch program.
        """
        raise NotImplementedError("Batch programs are no longer supported.")  # pragma: no cover

    batch_size = duet.sync(batch_size_async)

    async def delete_async(self, delete_jobs: bool = False) -> None:
        """Deletes a previously created quantum program.

        Params:
            delete_jobs: If True will delete all the program's jobs, other this
                will fail if the program contains any jobs.
        """
        await self.context.client.delete_program_async(
            self.project_id, self.program_id, delete_jobs=delete_jobs
        )

    delete = duet.sync(delete_async)

    async def delete_job_async(self, job_id: str) -> None:
        """Deletes the job and result, if any."""
        await self.context.client.delete_job_async(self.project_id, self.program_id, job_id)

    delete_job = duet.sync(delete_job_async)

    def __str__(self) -> str:
        return f'EngineProgram(project_id=\'{self.project_id}\', program_id=\'{self.program_id}\')'


def _deserialize_program(code: any_pb2.Any) -> cirq.Circuit:
    import cirq_google.engine.engine as engine_base

    code_type = code.type_url[len(engine_base.TYPE_PREFIX) :]
    program = None
    if code_type == 'cirq.google.api.v1.Program' or code_type == 'cirq.api.google.v1.Program':
        raise ValueError('deserializing a v1 Program is not supported')
    elif code_type == 'cirq.google.api.v2.Program' or code_type == 'cirq.api.google.v2.Program':
        program = v2.program_pb2.Program.FromString(code.value)
    if program is not None:
        return circuit_serializer.CIRCUIT_SERIALIZER.deserialize(program)

    raise ValueError(f'unsupported program type: {code_type}')
