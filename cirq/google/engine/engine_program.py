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

from typing import Optional, Sequence, TYPE_CHECKING

from cirq import study
from cirq.google.engine import engine_job

if TYPE_CHECKING:
    import cirq.google.engine.engine as engine


class EngineProgram:
    """A program created via the Quantum Engine API.

    This program wraps a Circuit with additional metadata used to
    schedule against the devices managed by Quantum Engine.

    Attributes:
        resource_name: The full resource name of the engine program.
    """

    def __init__(self, resource_name: str, engine: 'engine.Engine') -> None:
        """A job submitted to the engine.

        Args:
            resource_name: The globally unique identifier for the program:
                `projects/project_id/programs/program_id`.
            engine: An Engine object associated with the same project as the
                program.
        """
        self.resource_name = resource_name
        self._engine = engine

    def run_sweep(
            self,
            *,  # Force keyword args.
            job_config: Optional['engine.JobConfig'] = None,
            params: study.Sweepable = None,
            repetitions: int = 1,
            priority: int = 500,
            processor_ids: Sequence[str] = ('xmonsim',)
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

        Returns:
            An EngineJob. If this is iterated over it returns a list of
            TrialResults, one for each parameter sweep.
        """
        return self._engine.create_job(program_name=self.resource_name,
                                       job_config=job_config,
                                       params=params,
                                       repetitions=repetitions,
                                       priority=priority,
                                       processor_ids=processor_ids)

    def run(
            self,
            *,  # Force keyword args.
            job_config: Optional['engine.JobConfig'] = None,
            param_resolver: study.ParamResolver = study.ParamResolver({}),
            repetitions: int = 1,
            priority: int = 50,
            processor_ids: Sequence[str] = ('xmonsim',)) -> study.TrialResult:
        """Runs the supplied Circuit via Quantum Engine.

        Args:
            program: A Quantum Engine-wrapped Circuit object. This
              may be generated with create_program() or get_program().
            program_id: A user-provided identifier for the program. This must be
                unique within the Google Cloud project being used. If this
                parameter is not provided, a random id of the format
                'prog-######' will be generated.
            program_id: A user-defined identifer for the program. This must be
              unique within the project specified on the Engine instance.
            job_config: Configures the names of programs and jobs.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            priority: The priority to run at, 0-100.
            processor_ids: The engine processors that should be candidates
                to run the program. Only one of these will be scheduled for
                execution.

        Returns:
            A single TrialResult for this run.
        """
        return list(
            self.run_sweep(job_config=job_config,
                           params=[param_resolver],
                           repetitions=repetitions,
                           priority=priority,
                           processor_ids=processor_ids))[0]

    def __str__(self):
        return 'EngineProgram({})'.format(self.resource_name)