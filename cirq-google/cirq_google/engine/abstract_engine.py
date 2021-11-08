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
from abc import ABC, abstractmethod
import datetime
from typing import Dict, List, Optional, Sequence, Set, Union

import cirq
from cirq_google.engine import abstract_program, abstract_processor
from cirq_google.engine.client import quantum
from cirq_google.serialization import Serializer


class AbstractEngine(ABC):
    """An abstract object representing a collection of quantum procexsors.

    Each processor within the AbstractEngine can be referenced by a string
    identifier through the get_processor interface.

    The Engine interface also includes convenience methods to access
    programs, jobs, and sampler.

    This is an abstract interface and inheritors must implement the abstract methods.

    """

    @abstractmethod
    def get_program(self, program_id: str) -> abstract_program.AbstractProgram:
        """Returns an exsiting AbstractProgram given an identifier.

        Args:
            program_id: Unique ID of the program within the parent project.

        Returns:
            An AbstractProgram object for the program.
        """

    @abstractmethod
    def list_programs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ) -> List[abstract_program.AbstractProgram]:
        """Returns a list of previously executed quantum programs.

        Args:
            created_after: retrieve programs that were created after this date
                or time.
            created_before: retrieve programs that were created after this date
                or time.
            has_labels: retrieve programs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using
                `{'color: red', 'shape:*'}`
        """

    @abstractmethod
    def list_jobs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
        execution_states: Optional[Set[quantum.enums.ExecutionStatus.State]] = None,
    ):
        """Returns the list of jobs in the project.

        All historical jobs can be retrieved using this method and filtering
        options are available too, to narrow down the search baesd on:
          * creation time
          * job labels
          * execution states

        Args:
            created_after: retrieve jobs that were created after this date
                or time.
            created_before: retrieve jobs that were created after this date
                or time.
            has_labels: retrieve jobs that have labels on them specified by
                this dict. If the value is set to `*`, filters having the label
                regardless of the label value will be filtered. For example, to
                query programs that have the shape label and have the color
                label with value red can be queried using

                {'color': 'red', 'shape':'*'}

            execution_states: retrieve jobs that have an execution state  that
                 is contained in `execution_states`. See
                 `quantum.enums.ExecutionStatus.State` enum for accepted values.
        """

    @abstractmethod
    def list_processors(self) -> Sequence[abstract_processor.AbstractProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Returns:
            A list of EngineProcessors to access status, device and calibration
            information.
        """

    @abstractmethod
    def get_processor(self, processor_id: str) -> abstract_processor.AbstractProcessor:
        """Returns an EngineProcessor for a Quantum Engine processor.

        Args:
            processor_id: The processor unique identifier.

        Returns:
            A EngineProcessor for the processor.
        """

    @abstractmethod
    def get_sampler(
        self, processor_id: Union[str, List[str]], gate_set: Serializer
    ) -> cirq.Sampler:
        """Returns a sampler backed by the engine.

        Args:
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.
            gate_set: Determines how to serialize circuits when requesting
                samples.
        """
