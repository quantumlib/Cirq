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

import abc
from typing import Dict, Iterator, List, Optional, overload, Sequence, Tuple, TYPE_CHECKING

import duet

import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.engine_result import EngineResult

if TYPE_CHECKING:
    import datetime
    import cirq_google.engine.calibration as calibration
    import cirq_google.engine.calibration_result as calibration_result
    import cirq_google.engine.abstract_engine as abstract_engine
    import cirq_google.engine.abstract_processor as abstract_processor
    import cirq_google.engine.abstract_program as abstract_program


class AbstractJob(abc.ABC):
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

    @abc.abstractmethod
    def engine(self) -> 'abstract_engine.AbstractEngine':
        """Returns the parent `AbstractEngine` object."""

    @abc.abstractmethod
    def id(self) -> str:
        """Returns the id of this job."""

    @abc.abstractmethod
    def program(self) -> 'abstract_program.AbstractProgram':
        """Returns the parent `AbstractProgram`object."""

    @abc.abstractmethod
    def create_time(self) -> 'datetime.datetime':
        """Returns when the job was created."""

    @abc.abstractmethod
    def update_time(self) -> 'datetime.datetime':
        """Returns when the job was last updated."""

    @abc.abstractmethod
    def description(self) -> str:
        """Returns the description of the job."""

    @abc.abstractmethod
    def set_description(self, description: str) -> 'AbstractJob':
        """Sets the description of the job.

        Params:
            description: The new description for the job.

        Returns:
             This `AbstractJob`.
        """

    @abc.abstractmethod
    def labels(self) -> Dict[str, str]:
        """Returns the labels of the job."""

    @abc.abstractmethod
    def set_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        """Sets (overwriting) the labels for a previously created quantum job.

        Params:
            labels: The entire set of new job labels.

        Returns:
             This `AbstractJob`.
        """

    @abc.abstractmethod
    def add_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        """Adds new labels to a previously created quantum job.

        Params:
            labels: New labels to add to the existing job labels.

        Returns:
             This `AbstractJob`.
        """

    @abc.abstractmethod
    def remove_labels(self, keys: List[str]) -> 'AbstractJob':
        """Removes labels with given keys.

        Params:
            label_keys: Label keys to remove from the existing job labels.

        Returns:
            This `AbstractJob`.
        """

    @abc.abstractmethod
    def processor_ids(self) -> List[str]:
        """Returns the processor ids provided when the job was created."""

    @abc.abstractmethod
    def execution_status(self) -> quantum.ExecutionStatus.State:
        """Return the execution status of the job."""

    @abc.abstractmethod
    def failure(self) -> Optional[Tuple[str, str]]:
        """Return failure code and message of the job if present."""

    @abc.abstractmethod
    def get_repetitions_and_sweeps(self) -> Tuple[int, List[cirq.Sweep]]:
        """Returns the repetitions and sweeps for the job.

        Returns:
            A tuple of the repetition count and list of sweeps.
        """

    @abc.abstractmethod
    def get_processor(self) -> Optional['abstract_processor.AbstractProcessor']:
        """Returns the AbstractProcessor for the processor the job is/was run on,
        if available, else None."""

    @abc.abstractmethod
    def get_calibration(self) -> Optional['calibration.Calibration']:
        """Returns the recorded calibration at the time when the job was run, if
        one was captured, else None."""

    @abc.abstractmethod
    def cancel(self) -> Optional[bool]:
        """Cancel the job."""

    @abc.abstractmethod
    def delete(self) -> Optional[bool]:
        """Deletes the job and result, if any."""

    @abc.abstractmethod
    async def batched_results_async(self) -> Sequence[Sequence[EngineResult]]:
        """Returns the job results, blocking until the job is complete.

        This method is intended for batched jobs.  Instead of flattening
        results into a single list, this will return a List[Result]
        for each circuit in the batch.
        """

    batched_results = duet.sync(batched_results_async)

    @abc.abstractmethod
    async def results_async(self) -> Sequence[EngineResult]:
        """Returns the job results, blocking until the job is complete."""

    results = duet.sync(results_async)

    @abc.abstractmethod
    async def calibration_results_async(self) -> Sequence['calibration_result.CalibrationResult']:
        """Returns the results of a run_calibration() call.

        This function will fail if any other type of results were returned.
        """

    calibration_results = duet.sync(calibration_results_async)

    def __iter__(self) -> Iterator[cirq.Result]:
        yield from self.results()

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, item: int) -> cirq.Result:
        pass

    @overload
    def __getitem__(self, item: slice) -> Sequence[cirq.Result]:
        pass

    def __getitem__(self, item):
        return self.results()[item]

    # pylint: enable=function-redefined

    def __len__(self) -> int:
        return len(self.results())
