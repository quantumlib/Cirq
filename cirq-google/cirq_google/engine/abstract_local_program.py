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
import copy
import datetime
from typing import Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union

import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_program import AbstractProgram

if TYPE_CHECKING:
    from cirq_google.engine.abstract_local_engine import AbstractLocalEngine
    from cirq_google.engine.abstract_local_job import AbstractLocalJob


class AbstractLocalProgram(AbstractProgram):
    """A quantum program designed for local in-memory computation.

    This implements all the methods in `AbstractProgram` using
    in-memory objects.  Labels, descriptions, and time are all
    stored using dictionaries.

    This is a partially implemented instance.  Inheritors will still
    need to implement abstract methods.
    """

    def __init__(self, circuits: List[cirq.Circuit], engine: 'AbstractLocalEngine'):
        if not circuits:
            raise ValueError('No circuits provided to program.')
        self._create_time = datetime.datetime.now()
        self._update_time = datetime.datetime.now()
        self._description = ''
        self._labels: Dict[str, str] = {}
        self._engine = engine
        self._jobs: Dict[str, AbstractLocalJob] = {}
        self._circuits = circuits

    def engine(self) -> 'AbstractLocalEngine':
        """Returns the parent Engine object.

        Returns:
            The program's parent Engine.
        """
        return self._engine

    def add_job(self, job_id: str, job: 'AbstractLocalJob') -> None:
        self._jobs[job_id] = job

    def get_job(self, job_id: str) -> 'AbstractLocalJob':
        """Returns an AbstractLocalJob for an existing Quantum Engine job.

        Args:
            job_id: Unique ID of the job within the parent program.

        Returns:
            A AbstractLocalJob for this program.

        Raises:
            KeyError: if job is not found.
        """
        if job_id in self._jobs:
            return self._jobs[job_id]
        raise KeyError(f'job {job_id} not found')

    def list_jobs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
        execution_states: Optional[Set[quantum.ExecutionStatus.State]] = None,
    ) -> Sequence['AbstractLocalJob']:
        """Returns the list of jobs for this program.

        Args:
            created_after: retrieve jobs that were created after this date
                or time.
            created_before: retrieve jobs that were created before this date
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
        job_list = []
        for job in self._jobs.values():
            if created_before and job.create_time() > created_before:
                continue
            if created_after and job.create_time() < created_after:
                continue
            if execution_states:
                if job.execution_status() not in execution_states:
                    continue
            if has_labels:
                job_labels = job.labels()
                if not all(
                    label in job_labels and job_labels[label] == has_labels[label]
                    for label in has_labels
                ):
                    continue
            job_list.append(job)
        return job_list

    def create_time(self) -> 'datetime.datetime':
        """Returns when the program was created."""
        return self._create_time

    def update_time(self) -> 'datetime.datetime':
        """Returns when the program was last updated."""
        return self._update_time

    def description(self) -> str:
        """Returns the description of the program."""
        return self._description

    def set_description(self, description: str) -> 'AbstractProgram':
        """Sets the description of the program.

        Params:
            description: The new description for the program.

        Returns:
             This AbstractProgram.
        """
        self._description = description
        return self

    def labels(self) -> Dict[str, str]:
        """Returns the labels of the program."""
        return copy.copy(self._labels)

    def set_labels(self, labels: Dict[str, str]) -> 'AbstractProgram':
        """Sets (overwriting) the labels for a previously created quantum
        program.

        Params:
            labels: The entire set of new program labels.

        Returns:
             This AbstractProgram.
        """
        self._labels = copy.copy(labels)
        return self

    def add_labels(self, labels: Dict[str, str]) -> 'AbstractProgram':
        """Adds new labels to a previously created quantum program.

        Params:
            labels: New labels to add to the existing program labels.

        Returns:
             This AbstractProgram.
        """
        for key in labels:
            self._labels[key] = labels[key]
        return self

    def remove_labels(self, keys: List[str]) -> 'AbstractProgram':
        """Removes labels with given keys from the labels of a previously
        created quantum program.

        Params:
            label_keys: Label keys to remove from the existing program labels.

        Returns:
             This AbstractProgram.
        """
        for key in keys:
            del self._labels[key]
        return self

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
        if program_num:
            return self._circuits[program_num]
        return self._circuits[0]

    def batch_size(self) -> int:
        """Returns the number of programs in a batch program."""
        return len(self._circuits)
