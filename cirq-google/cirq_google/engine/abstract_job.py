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
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, overload, Tuple, TYPE_CHECKING

from cirq_google.engine.client import quantum
from cirq_google.engine.calibration_result import CalibrationResult

import cirq

if TYPE_CHECKING:
    import datetime
    from cirq_google.engine.calibration import Calibration
    from cirq_google.engine.abstract_engine import AbstractEngine
    from cirq_google.engine.abstract_processor import AbstractProcessor
    from cirq_google.engine.abstract_program import AbstractProgram


class AbstractJob(ABC):
    """An abstract object representing a quantum job execution.

    This represents the state of a possibly asynchronous Job being
    executed by a simulator, the cloud Engine service, or other means.

    This is an abstract interface that implementers of services or mocks
    should implement.  It generally represents the execution of a circuit
    using a set of parameters called a sweep.  It can also represent the
    execution of a batch job (a list of circuit/sweep pairs) or the
    execution of a calibration request.

    This job may be in a variety of states. It may be scheduling, it may be
    executing on a machine, or it may have entered a terminal state
    (either succeeding or failing).

    `AbstractJob`s can be iterated over, returning `Result`s. These
    `Result`s can also be accessed by index. Note that this will block
    until the results are returned.

    """

    @abstractmethod
    def engine(self) -> 'AbstractEngine':
        """Returns the parent `AbstractEngine` object."""

    @abstractmethod
    def id(self) -> str:
        """Returns the id of this job."""

    @abstractmethod
    def program(self) -> 'AbstractProgram':
        """Returns the parent `AbstractProgram`object."""

    @abstractmethod
    def create_time(self) -> 'datetime.datetime':
        """Returns when the job was created."""

    @abstractmethod
    def update_time(self) -> 'datetime.datetime':
        """Returns when the job was last updated."""

    @abstractmethod
    def description(self) -> str:
        """Returns the description of the job."""

    @abstractmethod
    def set_description(self, description: str) -> 'AbstractJob':
        """Sets the description of the job.

        Params:
            description: The new description for the job.

        Returns:
             This `AbstractJob`.
        """

    @abstractmethod
    def labels(self) -> Dict[str, str]:
        """Returns the labels of the job."""

    @abstractmethod
    def set_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        """Sets (overwriting) the labels for a previously created quantum job.

        Params:
            labels: The entire set of new job labels.

        Returns:
             This `AbstractJob`.
        """

    @abstractmethod
    def add_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        """Adds new labels to a previously created quantum job.

        Params:
            labels: New labels to add to the existing job labels.

        Returns:
             This `AbstractJob`.
        """

    @abstractmethod
    def remove_labels(self, keys: List[str]) -> 'AbstractJob':
        """Removes labels with given keys.

        Params:
            label_keys: Label keys to remove from the existing job labels.

        Returns:
            This `AbstractJob`.
        """

    @abstractmethod
    def processor_ids(self) -> List[str]:
        """Returns the processor ids provided when the job was created."""

    @abstractmethod
    def execution_status(self) -> quantum.enums.ExecutionStatus.State:
        """Return the execution status of the job."""

    @abstractmethod
    def failure(self) -> Optional[Tuple[str, str]]:
        """Return failure code and message of the job if present."""

    @abstractmethod
    def get_repetitions_and_sweeps(self) -> Tuple[int, List[cirq.Sweep]]:
        """Returns the repetitions and sweeps for the job.

        Returns:
            A tuple of the repetition count and list of sweeps.
        """

    @abstractmethod
    def get_processor(self) -> 'Optional[AbstractProcessor]':
        """Returns the AbstractProcessor for the processor the job is/was run on,
        if available, else None."""

    @abstractmethod
    def get_calibration(self) -> Optional['Calibration']:
        """Returns the recorded calibration at the time when the job was run, if
        one was captured, else None."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the job."""

    @abstractmethod
    def delete(self) -> None:
        """Deletes the job and result, if any."""

    @abstractmethod
    def batched_results(self) -> List[List[cirq.Result]]:
        """Returns the job results, blocking until the job is complete.

        This method is intended for batched jobs.  Instead of flattening
        results into a single list, this will return a List[Result]
        for each circuit in the batch.
        """

    @abstractmethod
    def results(self) -> List[cirq.Result]:
        """Returns the job results, blocking until the job is complete."""

    @abstractmethod
    def calibration_results(self) -> List[CalibrationResult]:
        """Returns the results of a run_calibration() call.

        This function will fail if any other type of results were returned.
        """

    def __iter__(self) -> Iterator[cirq.Result]:
        return iter(self.results())

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, item: int) -> cirq.Result:
        pass

    @overload
    def __getitem__(self, item: slice) -> List[cirq.Result]:
        pass

    def __getitem__(self, item):
        return self.results()[item]

    # pylint: enable=function-redefined

    def __len__(self) -> int:
        return len(self.results())
