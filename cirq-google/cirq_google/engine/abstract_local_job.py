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
"""A helper for jobs that have been created on the Quantum Engine."""
import copy
import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cirq
from cirq_google.engine import calibration
from cirq_google.engine.abstract_job import AbstractJob

if TYPE_CHECKING:
    from cirq_google.engine.abstract_local_engine import AbstractLocalEngine
    from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
    from cirq_google.engine.abstract_local_program import AbstractLocalProgram


class AbstractLocalJob(AbstractJob):
    """A job that handles labels and descriptions locally in-memory.

        This class is designed to make writing custom AbstractJob objects
        that function in-memory easier.  This class will handle basic functionality
        expected to be common across all local implementations.

        Implementors of this class should write the following functions:
          - Status functions: execution_status, failure
          - Action functions: cancel, delete
          - Result functions: results, batched_results, calibration_results
    `
        Attributes:
          processor_ids: A string list of processor ids that this job can be run on.
          processor_id: If provided, the processor id that the job was run on.
              If not provided, assumed to be the first element of processor_ids
          parent_program: Program containing this job
          repetitions: number of repetitions for each parameter set
          sweeps: list of Sweeps that this job should iterate through.
    """

    def __init__(
        self,
        *,
        job_id: str,
        parent_program: 'AbstractLocalProgram',
        repetitions: int,
        sweeps: List[cirq.Sweep],
        processor_id: str = '',
    ):
        self._id = job_id
        self._processor_id = processor_id
        self._parent_program = parent_program
        self._repetitions = repetitions
        self._sweeps = sweeps
        self._create_time = datetime.datetime.now()
        self._update_time = datetime.datetime.now()
        self._description = ''
        self._labels: Dict[str, str] = {}

    def engine(self) -> 'AbstractLocalEngine':
        """Returns the parent program's `AbstractEngine` object."""
        return self._parent_program.engine()

    def id(self) -> str:
        """Returns the identifier of this job."""
        return self._id

    def program(self) -> 'AbstractLocalProgram':
        """Returns the parent `AbstractLocalProgram` object."""
        return self._parent_program

    def create_time(self) -> 'datetime.datetime':
        """Returns when the job was created."""
        return self._create_time

    def update_time(self) -> 'datetime.datetime':
        """Returns when the job was last updated."""
        return self._update_time

    def description(self) -> str:
        """Returns the description of the job."""
        return self._description

    def set_description(self, description: str) -> 'AbstractJob':
        """Sets the description of the job.

        Params:
            description: The new description for the job.

        Returns:
             This AbstractJob.
        """
        self._description = description
        self._update_time = datetime.datetime.now()
        return self

    def labels(self) -> Dict[str, str]:
        """Returns the labels of the job."""
        return copy.copy(self._labels)

    def set_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        """Sets (overwriting) the labels for a previously created quantum job.

        Params:
            labels: The entire set of new job labels.

        Returns:
             This AbstractJob.
        """
        self._labels = copy.copy(labels)
        self._update_time = datetime.datetime.now()
        return self

    def add_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        """Adds new labels to a previously created quantum job.

        Params:
            labels: New labels to add to the existing job labels.

        Returns:
             This AbstractJob.
        """
        self._update_time = datetime.datetime.now()
        for key in labels:
            self._labels[key] = labels[key]
        return self

    def remove_labels(self, keys: List[str]) -> 'AbstractJob':
        """Removes labels with given keys from the labels of a previously
        created quantum job.

        Params:
            label_keys: Label keys to remove from the existing job labels.

        Returns:
            This AbstractJob.
        """
        self._update_time = datetime.datetime.now()
        for key in keys:
            del self._labels[key]
        return self

    def processor_ids(self) -> List[str]:
        """Returns the processor ids provided when the job was created."""
        return [self._processor_id]

    def get_repetitions_and_sweeps(self) -> Tuple[int, List[cirq.Sweep]]:
        """Returns the repetitions and sweeps for the job.

        Returns:
            A tuple of the repetition count and list of sweeps.
        """
        return (self._repetitions, self._sweeps)

    def get_processor(self) -> 'AbstractLocalProcessor':
        """Returns the AbstractProcessor for the processor the job is/was run on,
        if available, else None."""
        return self.engine().get_processor(self._processor_id)

    def get_calibration(self) -> Optional[calibration.Calibration]:
        """Returns the recorded calibration at the time when the job was created,
        from the parent Engine object."""
        return self.get_processor().get_latest_calibration(int(self._create_time.timestamp()))
