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
"""Interface for Engine objects.

This class is an abstract class which all Engine implementations
(production API or locally simulated) should follow.
"""

import abc
import datetime
from typing import Dict, List, Optional, Sequence, Set, Union

import cirq
from cirq_google.cloud import quantum
from cirq_google.engine import abstract_job, abstract_processor, abstract_program

VALID_DATE_TYPE = Union[datetime.datetime, datetime.date]


class AbstractEngine(abc.ABC):
    """An abstract object representing a collection of quantum processors.

    Each processor within the AbstractEngine can be referenced by a string
    identifier through the get_processor interface.

    The Engine interface also includes convenience methods to access
    programs, jobs, and sampler.

    This is an abstract interface and inheritors must implement the abstract methods.

    """

    @abc.abstractmethod
    def get_program(self, program_id: str) -> abstract_program.AbstractProgram:
        """Returns an existing AbstractProgram given an identifier.

        Args:
            program_id: Unique ID of the program.

        Returns:
            An AbstractProgram object for the program.
        """

    @abc.abstractmethod
    def list_programs(
        self,
        created_before: Optional[VALID_DATE_TYPE] = None,
        created_after: Optional[VALID_DATE_TYPE] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ) -> List[abstract_program.AbstractProgram]:
        """Returns a list of previously executed quantum programs.

        Args:
            created_after: retrieve programs that were created after this date
                or time.
            created_before: retrieve programs that were created before this date
                or time.
            has_labels: retrieve programs that have labels on them specified by
                this dict. If the value is set to `*`, programs having the label
                regardless of the label value will be returned. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using
                `{'color': 'red', 'shape': '*'}`
        """

    @abc.abstractmethod
    def list_jobs(
        self,
        created_before: Optional[VALID_DATE_TYPE] = None,
        created_after: Optional[VALID_DATE_TYPE] = None,
        has_labels: Optional[Dict[str, str]] = None,
        execution_states: Optional[Set[quantum.ExecutionStatus.State]] = None,
    ) -> List[abstract_job.AbstractJob]:
        """Returns the list of jobs that match the specified criteria.

        All historical jobs can be retrieved using this method and filtering
        options are available too, to narrow down the search based on:
          * creation time
          * job labels
          * execution states

        Args:
            created_after: retrieve jobs that were created after this date
                or time.
            created_before: retrieve jobs that were created before this date
                or time.
            has_labels: retrieve jobs that have labels on them specified by
                this dict. If the value is set to `*`, jobs having the label
                regardless of the label value will be returned. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}

            execution_states: retrieve jobs that have an execution state  that
                 is contained in `execution_states`. See
                 `quantum.ExecutionStatus.State` enum for accepted values.
        """

    @abc.abstractmethod
    def list_processors(self) -> Sequence[abstract_processor.AbstractProcessor]:
        """Returns all processors in this engine visible to the user."""

    @abc.abstractmethod
    def get_processor(self, processor_id: str) -> abstract_processor.AbstractProcessor:
        """Returns an EngineProcessor for a Quantum Engine processor.

        Args:
            processor_id: The processor unique identifier.

        Returns:
            A EngineProcessor for the processor.
        """

    @abc.abstractmethod
    def get_sampler(self, processor_id: Union[str, List[str]]) -> cirq.Sampler:
        """Returns a sampler backed by the engine.

        Args:
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.
        """
