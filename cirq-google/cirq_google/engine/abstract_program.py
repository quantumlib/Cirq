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
"""An interface for quantum programs.

The quantum program represents a circuit (or other execution) that,
when combined with a run context, will become a quantum job.
"""


import abc
import datetime
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union
import cirq
from cirq_google.engine.client import quantum

if TYPE_CHECKING:
    import cirq_google.engine.abstract_job as abstract_job
    import cirq_google.engine.abstract_engine as abstract_engine


class AbstractProgram(abc.ABC):
    """An abstract object representing a quantum program.

    This program generally wraps a `Circuit` with additional metadata.
    When combined with an appropriate RunContext, this becomes a
    Job that can run on either an Engine service or simulator.
    Programs can also be a batch (list of circuits) or calibration
    requests.

    This is an abstract class that inheritors should implement.
    """

    @abc.abstractmethod
    def engine(self) -> 'abstract_engine.AbstractEngine':
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.
        """

    @abc.abstractmethod
    def get_job(self, job_id: str) -> 'abstract_job.AbstractJob':
        """Returns an AbstractJob for an existing id.

        Args:
            job_id: Unique ID of the job within the parent program.

        Returns:
            A AbstractJob for this program.
        """

    @abc.abstractmethod
    def list_jobs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
        execution_states: Optional[Set[quantum.enums.ExecutionStatus.State]] = None,
    ) -> Sequence['abstract_job.AbstractJob']:
        """Returns the list of jobs for this program.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            created_after: retrieve jobs that were created after this date
                or time.
            created_before: retrieve jobs that were created after this date
                or time.
            has_labels: retrieve jobs that have labels on them specified by
                this dict. If the value is set to `*`, jobs having the label
                regardless of the label value will be returned. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}

            execution_states: retrieve jobs that have an execution state  that
                is contained in `execution_states`. See
                `quantum.enums.ExecutionStatus.State` enum for accepted values.

        Returns:
            A sequence of `AbstractJob` objects that satisfy the constraints.
        """

    @abc.abstractmethod
    def create_time(self) -> 'datetime.datetime':
        """Returns when the program was created."""

    @abc.abstractmethod
    def update_time(self) -> 'datetime.datetime':
        """Returns when the program was last updated."""

    @abc.abstractmethod
    def description(self) -> str:
        """Returns the description of the program."""

    @abc.abstractmethod
    def set_description(self, description: str) -> 'AbstractProgram':
        """Sets the description of the program.

        Params:
            description: The new description for the program.

        Returns:
             This AbstractProgram.
        """

    @abc.abstractmethod
    def labels(self) -> Dict[str, str]:
        """Returns the labels of the program."""

    @abc.abstractmethod
    def set_labels(self, labels: Dict[str, str]) -> 'AbstractProgram':
        """Sets (overwriting) the labels for a previously created quantum program.

        Params:
            labels: The entire set of new program labels.

        Returns:
             This AbstractProgram.
        """

    @abc.abstractmethod
    def add_labels(self, labels: Dict[str, str]) -> 'AbstractProgram':
        """Adds new labels to a previously created quantum program.

        Params:
            labels: New labels to add to the existing program labels.

        Returns:
             This AbstractProgram.
        """

    @abc.abstractmethod
    def remove_labels(self, keys: List[str]) -> 'AbstractProgram':
        """Removes labels with given keys from the labels of a previously
        created quantum program.

        Params:
            label_keys: Label keys to remove from the existing program labels.

        Returns:
             This AbstractProgram.
        """

    @abc.abstractmethod
    def get_circuit(self, program_num: Optional[int] = None) -> cirq.Circuit:
        """Returns the cirq Circuit for the program. This is only
        supported if the program was created with the V2 protos.

        Args:
            program_num: if this is a batch program, the index of the circuit in
                the batch.  This argument is zero-indexed. Negative values
                indexing from the end of the list.

        Returns:
            The program's cirq Circuit.
        """

    @abc.abstractmethod
    def batch_size(self) -> int:
        """Returns the number of programs in a batch program.

        Raises:
            ValueError: if the program created was not a batch program.
        """

    @abc.abstractmethod
    def delete(self, delete_jobs: bool = False) -> None:
        """Deletes a previously created quantum program.

        Params:
            delete_jobs: If True will delete all the program's jobs, other this
                will fail if the program contains any jobs.
        """

    @abc.abstractmethod
    def delete_job(self, job_id: str) -> None:
        """Removes a child job from this program."""
