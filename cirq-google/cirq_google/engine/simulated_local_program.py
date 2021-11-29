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
from typing import Optional, TYPE_CHECKING

from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.local_simulation_type import LocalSimulationType

if TYPE_CHECKING:
    from cirq_google.engine.abstract_job import AbstractJob
    from cirq_google.engine.abstract_engine import AbstractEngine
    from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor


class SimulatedLocalProgram(AbstractLocalProgram):
    """A program backed by a (local) sampler.

    This class functions as a parent class for a `SimulatedLocalJob`
    object.
    """

    def __init__(
        self,
        *args,
        program_id: str,
        simulation_type: LocalSimulationType = LocalSimulationType.SYNCHRONOUS,
        processor: Optional['SimulatedLocalProcessor'] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._id = program_id
        self._processor = processor

    def delete(self, delete_jobs: bool = False) -> None:
        if self._processor:
            self._processor.remove_program(self._id)
        if delete_jobs:
            for job in list(self._jobs.values()):
                job.delete()

    def delete_job(self, job_id: str) -> None:
        del self._jobs[job_id]

    def id(self) -> str:
        return self._id
