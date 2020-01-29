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
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from cirq import study
from cirq.google.engine import engine_job
from cirq.google.engine.client.quantum import types as qtypes
from cirq.google import gate_sets
from cirq.google.api import v1, v2
import cirq.google.engine.engine as engine_base

if TYPE_CHECKING:
    from cirq import Circuit


class EngineProgram:
    """A program created via the Quantum Engine API.

    This program wraps a Circuit with additional metadata used to
    schedule against the devices managed by Quantum Engine.

    Attributes:
        project_id: A project_id of the parent Google Cloud Project.
        program_id: Unique ID of the program within the parent project.
    """

    def __init__(self, program_id: str, engine: 'engine_base.Engine',
                 program: qtypes.QuantumProgram) -> None:
        """A job submitted to the engine.

        Args:
            program_id: Unique ID of the program within the parent project.
            engine: The parent Engine object.
            program: The optional current program state.
        """
        self.project_id = engine.project_id
        self.program_id = program_id
        self._engine = engine
        self._program = program

    def run_sweep(
            self,
            *,  # Force keyword args.
            job_config: 'Optional[engine_base.JobConfig]' = None,
            params: study.Sweepable = None,
            repetitions: int = 1,
            priority: int = 500,
            processor_ids: Sequence[str] = ('xmonsim',),
            description: Optional[str] = None,
            labels: Optional[Dict[str, str]] = None,
    ) -> engine_job.EngineJob:
        """Runs the program on the QuantumEngine.

        In contrast to run, this runs across multiple parameter sweeps, and
        does not block until a result is returned.

        Args:
            job_config: Configures optional job parameters.
            params: Parameters to run with the program.
            repetitions: The number of circuit repetitions to run.
            priority: The priority to run at, 0-100.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            description: An optional description to set on the job.
            labels: Optional set of label to set on the job.

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.
        """
        # Check program to run and program parameters.
        if not 0 <= priority < 1000:
            raise ValueError('priority must be between 0 and 1000')

        job_config = engine_base.implied_job_config(job_config)
        sweeps = study.to_sweeps(params or study.ParamResolver({}))
        run_context = self._serialize_run_context(sweeps, repetitions)

        created_job_id, job = self._engine.client.create_job(
            project_id=self.project_id,
            program_id=self.program_id,
            job_id=job_config.job_id,
            processor_ids=processor_ids,
            run_context=run_context,
            priority=priority,
            description=description,
            labels=labels)
        return engine_job.EngineJob(created_job_id, self, job)

    def run(
            self,
            *,  # Force keyword args.
            job_config: 'Optional[engine_base.JobConfig]' = None,
            param_resolver: study.ParamResolver = study.ParamResolver({}),
            repetitions: int = 1,
            priority: int = 50,
            processor_ids: Sequence[str] = ('xmonsim',),
            description: Optional[str] = None,
            labels: Optional[Dict[str, str]] = None,
    ) -> study.TrialResult:
        """Runs the supplied Circuit via Quantum Engine.

        Args:
            job_config: Configures the names of programs and jobs.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            priority: The priority to run at, 0-100.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            description: An optional description to set on the job.
            labels: Optional set of label to set on the job.

        Returns:
            A single TrialResult for this run.
        """
        return list(
            self.run_sweep(job_config=job_config,
                           params=[param_resolver],
                           repetitions=repetitions,
                           priority=priority,
                           processor_ids=processor_ids,
                           description=description,
                           labels=labels))[0]

    def get_job(self,
                job_id: str,
                project_id: Optional[str] = None,
                program_id: Optional[str] = None) -> engine_job.EngineJob:
        """Creates an EngineJob for an existing Quantum Engine job.

        Args:
            job_id: Unique ID of the job within the parent program.
            project_id: The project id for the project containing the program.
                If provided will be checked against the project id of the
                EngineProgram.
            program_id: Unique ID of the program within the parent project.
                If provided will be checked against the program id of the
                EngineProgram.

        Returns:
            A EngineJob for the job.
        """
        if project_id and project_id != self.project_id:
            raise ValueError(
                'EngineProgram project id {} does not match given project_id {}'
                .format(self.project_id, project_id))

        if program_id and program_id != self.program_id:
            raise ValueError(
                'EngineProgram program id {} does not match given program_id {}'
                .format(self.program_id, program_id))

        job = self.engine().client.get_job(self.project_id, self.program_id,
                                           job_id, False)
        return engine_job.EngineJob(job_id, self, job)

    def engine(self) -> 'engine_base.Engine':
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.
        """
        return self._engine

    def _serialize_run_context(
            self,
            sweeps: List[study.Sweep],
            repetitions: int,
    ) -> qtypes.any_pb2.Any:
        context = qtypes.any_pb2.Any()
        proto_version = self._engine.proto_version
        if proto_version == engine_base.ProtoVersion.V1:
            context.Pack(
                v1.program_pb2.RunContext(parameter_sweeps=[
                    v1.sweep_to_proto(sweep, repetitions) for sweep in sweeps
                ]))
        elif proto_version == engine_base.ProtoVersion.V2:
            run_context = v2.run_context_pb2.RunContext()
            for sweep in sweeps:
                sweep_proto = run_context.parameter_sweeps.add()
                sweep_proto.repetitions = repetitions
                v2.sweep_to_proto(sweep, out=sweep_proto.sweep)

            context.Pack(run_context)
        else:
            raise ValueError(
                'invalid run context proto version: {}'.format(proto_version))
        return context

    def get_circuit(self) -> 'Circuit':
        """Returns the cirq Circuit for the Quantum Engine program. This is only
        supported if the program was created with the V2 protos.

        Returns:
            The program's cirq Circuit.
        """
        if not self._program.HasField('code'):
            self._program = self._engine.client.get_program(
                self.project_id, self.program_id, True)
        return self._deserialize_program(self._program.code)

    def _deserialize_program(self, code: qtypes.any_pb2.Any) -> 'Circuit':
        code_type = code.type_url[len(engine_base.TYPE_PREFIX):]
        if (code_type == 'cirq.google.api.v1.Program' or
                code_type == 'cirq.api.google.v1.Program'):
            raise ValueError('deserializing a v1 Program is not supported')
        if (code_type == 'cirq.google.api.v2.Program' or
                code_type == 'cirq.api.google.v2.Program'):
            program = v2.program_pb2.Program()
            program.ParseFromString(code.value)
            gate_set_map = {
                g.gate_set_name: g for g in gate_sets.GOOGLE_GATESETS
            }
            try:
                return gate_set_map[program.language.gate_set].deserialize(
                    program)
            except KeyError:
                raise ValueError('unsupported gateset: {}'.format(
                    program.language.gate_set))
        raise ValueError('unsupported program type: {}'.format(code_type))

    def description(self) -> str:
        """Returns the description of the program.

        Returns:
             The current description of the program.
        """
        return self._program.description

    def set_description(self, description: str) -> 'EngineProgram':
        """Sets the description of the program.

        Params:
            description: The new description for the program.

        Returns:
             This EngineProgram.
        """
        self._program = self._engine.client.set_program_description(
            self.project_id, self.program_id, description)
        return self

    def labels(self) -> Dict[str, str]:
        """Returns the labels of the program.

        Returns:
             The current labels of the program.
        """
        return self._program.labels

    def set_labels(self, labels: Dict[str, str]) -> 'EngineProgram':
        """Sets (overwriting) the labels for a previously created quantum
        program.

        Params:
            labels: The entire set of new program labels.

        Returns:
             This EngineProgram.
        """
        self._program = self._engine.client.set_program_labels(
            self.project_id, self.program_id, labels)
        return self

    def add_labels(self, labels: Dict[str, str]) -> 'EngineProgram':
        """Adds new labels to a previously created quantum program.

        Params:
            labels: New labels to add to the existing program labels.

        Returns:
             This EngineProgram.
        """
        self._program = self._engine.client.add_program_labels(
            self.project_id, self.program_id, labels)
        return self

    def remove_labels(self, keys: List[str]) -> 'EngineProgram':
        """Removes labels with given keys from the labels of a previously
        created quantum program.

        Params:
            label_keys: Label keys to remove from the existing program labels.

        Returns:
             This EngineProgram.
        """
        self._program = self._engine.client.remove_program_labels(
            self.project_id, self.program_id, keys)
        return self

    def delete(self, delete_jobs: bool = False) -> None:
        """Deletes a previously created quantum program.

        Params:
            delete_jobs: If True will delete all the program's jobs, other this
                will fail if the program contains any jobs.
        """
        self._engine.client.delete_program(self.project_id,
                                           self.program_id,
                                           delete_jobs=delete_jobs)

    def __str__(self):
        return 'EngineProgram(project_id=\'{}\', program_id=\'{}\')'.format(
            self.project_id, self.program_id)
