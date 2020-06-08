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
from cirq.google.engine.client.quantum import types as qtypes
from cirq.google import gate_sets
from cirq.google.api import v1, v2
from cirq.google.engine import engine_job

if TYPE_CHECKING:
    import datetime
    import cirq.google.engine.engine as engine_base
    from cirq import Circuit


class EngineProgram:
    """A program created via the Quantum Engine API.

    This program wraps a Circuit with additional metadata used to
    schedule against the devices managed by Quantum Engine.

    Attributes:
        project_id: A project_id of the parent Google Cloud Project.
        program_id: Unique ID of the program within the parent project.
    """

    def __init__(self,
                 project_id: str,
                 program_id: str,
                 context: 'engine_base.EngineContext',
                 _program: Optional[qtypes.QuantumProgram] = None) -> None:
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

    def run_sweep(
            self,
            job_id: Optional[str] = None,
            params: study.Sweepable = None,
            repetitions: int = 1,
            processor_ids: Sequence[str] = ('xmonsim',),
            description: Optional[str] = None,
            labels: Optional[Dict[str, str]] = None,
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
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            description: An optional description to set on the job.
            labels: Optional set of labels to set on the job.

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.
        """
        import cirq.google.engine.engine as engine_base
        if not job_id:
            job_id = engine_base._make_random_id('job-')
        sweeps = study.to_sweeps(params or study.ParamResolver({}))
        run_context = self._serialize_run_context(sweeps, repetitions)

        created_job_id, job = self.context.client.create_job(
            project_id=self.project_id,
            program_id=self.program_id,
            job_id=job_id,
            processor_ids=processor_ids,
            run_context=run_context,
            description=description,
            labels=labels)
        return engine_job.EngineJob(self.project_id, self.program_id,
                                    created_job_id, self.context, job)

    def run(
            self,
            job_id: Optional[str] = None,
            param_resolver: study.ParamResolver = study.ParamResolver({}),
            repetitions: int = 1,
            processor_ids: Sequence[str] = ('xmonsim',),
            description: Optional[str] = None,
            labels: Optional[Dict[str, str]] = None,
    ) -> study.TrialResult:
        """Runs the supplied Circuit via Quantum Engine.

        Args:
            job_id: Optional job id to use. If this is not provided, a random id
                of the format 'job-################YYMMDD' will be generated,
                where # is alphanumeric and YYMMDD is the current year, month,
                and day.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.
            description: An optional description to set on the job.
            labels: Optional set of labels to set on the job.

        Returns:
            A single TrialResult for this run.
        """
        return list(
            self.run_sweep(job_id=job_id,
                           params=[param_resolver],
                           repetitions=repetitions,
                           processor_ids=processor_ids,
                           description=description,
                           labels=labels))[0]

    def _serialize_run_context(
            self,
            sweeps: List[study.Sweep],
            repetitions: int,
    ) -> qtypes.any_pb2.Any:
        import cirq.google.engine.engine as engine_base
        context = qtypes.any_pb2.Any()
        proto_version = self.context.proto_version
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

    def engine(self) -> 'engine_base.Engine':
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.
        """
        import cirq.google.engine.engine as engine_base
        return engine_base.Engine(self.project_id, context=self.context)

    def get_job(self, job_id: str) -> engine_job.EngineJob:
        """Returns an EngineJob for an existing Quantum Engine job.

        Args:
            job_id: Unique ID of the job within the parent program.

        Returns:
            A EngineJob for the job.
        """
        return engine_job.EngineJob(self.project_id, self.program_id, job_id,
                                    self.context)

    def _inner_program(self) -> qtypes.QuantumProgram:
        if not self._program:
            self._program = self.context.client.get_program(
                self.project_id, self.program_id, False)
        return self._program

    def create_time(self) -> 'datetime.datetime':
        """Returns when the program was created."""
        return self._inner_program().create_time.ToDatetime()

    def update_time(self) -> 'datetime.datetime':
        """Returns when the program was last updated."""
        self._program = self.context.client.get_program(self.project_id,
                                                        self.program_id, False)
        return self._program.update_time.ToDatetime()

    def description(self) -> str:
        """Returns the description of the program."""
        return self._inner_program().description

    def set_description(self, description: str) -> 'EngineProgram':
        """Sets the description of the program.

        Params:
            description: The new description for the program.

        Returns:
             This EngineProgram.
        """
        self._program = self.context.client.set_program_description(
            self.project_id, self.program_id, description)
        return self

    def labels(self) -> Dict[str, str]:
        """Returns the labels of the program."""
        return self._inner_program().labels

    def set_labels(self, labels: Dict[str, str]) -> 'EngineProgram':
        """Sets (overwriting) the labels for a previously created quantum
        program.

        Params:
            labels: The entire set of new program labels.

        Returns:
             This EngineProgram.
        """
        self._program = self.context.client.set_program_labels(
            self.project_id, self.program_id, labels)
        return self

    def add_labels(self, labels: Dict[str, str]) -> 'EngineProgram':
        """Adds new labels to a previously created quantum program.

        Params:
            labels: New labels to add to the existing program labels.

        Returns:
             This EngineProgram.
        """
        self._program = self.context.client.add_program_labels(
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
        self._program = self.context.client.remove_program_labels(
            self.project_id, self.program_id, keys)
        return self

    def get_circuit(self) -> 'Circuit':
        """Returns the cirq Circuit for the Quantum Engine program. This is only
        supported if the program was created with the V2 protos.

        Returns:
            The program's cirq Circuit.
        """
        if not self._program or not self._program.HasField('code'):
            self._program = self.context.client.get_program(
                self.project_id, self.program_id, True)
        return self._deserialize_program(self._program.code)

    @staticmethod
    def _deserialize_program(code: qtypes.any_pb2.Any) -> 'Circuit':
        import cirq.google.engine.engine as engine_base
        code_type = code.type_url[len(engine_base.TYPE_PREFIX):]
        if (code_type == 'cirq.google.api.v1.Program' or
                code_type == 'cirq.api.google.v1.Program'):
            raise ValueError('deserializing a v1 Program is not supported')
        if (code_type == 'cirq.google.api.v2.Program' or
                code_type == 'cirq.api.google.v2.Program'):
            program = v2.program_pb2.Program.FromString(code.value)
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

    def delete(self, delete_jobs: bool = False) -> None:
        """Deletes a previously created quantum program.

        Params:
            delete_jobs: If True will delete all the program's jobs, other this
                will fail if the program contains any jobs.
        """
        self.context.client.delete_program(self.project_id,
                                           self.program_id,
                                           delete_jobs=delete_jobs)

    def __str__(self) -> str:
        return (f'EngineProgram(project_id=\'{self.project_id}\', '
                f'program_id=\'{self.program_id}\')')
