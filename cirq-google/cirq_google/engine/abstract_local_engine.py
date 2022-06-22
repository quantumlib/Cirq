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
import datetime
from typing import Dict, List, Optional, Sequence, Set, Union

import cirq
from cirq_google.engine.abstract_job import AbstractJob
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_engine import AbstractEngine
from cirq_google.cloud import quantum


class AbstractLocalEngine(AbstractEngine):
    """Collection of processors that can execute quantum jobs.

    This class assumes that all processors are local.  Processors
    are given during initialization.  Program and job querying
    functionality is done by serially querying all child processors.

    """

    def __init__(self, processors: List[AbstractLocalProcessor]):
        for processor in processors:
            processor.set_engine(self)
        self._processors = {proc.processor_id: proc for proc in processors}

    def get_program(self, program_id: str) -> AbstractProgram:
        """Returns an exsiting AbstractProgram given an identifier.

        Iteratively checks each processor for the given id.

        Args:
            program_id: Unique ID of the program within the parent project.

        Returns:
            An AbstractProgram for the program.

        Raises:
            KeyError: if program does not exist
        """
        for processor in self._processors.values():
            try:
                return processor.get_program(program_id)
            except KeyError:
                continue
        raise KeyError(f'Program {program_id} does not exist')

    def list_programs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
    ) -> List[AbstractProgram]:
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
        valid_programs: List[AbstractProgram] = []
        for processor in self._processors.values():
            valid_programs.extend(
                processor.list_programs(
                    created_before=created_before,
                    created_after=created_after,
                    has_labels=has_labels,
                )
            )
        return valid_programs

    def list_jobs(
        self,
        created_before: Optional[Union[datetime.datetime, datetime.date]] = None,
        created_after: Optional[Union[datetime.datetime, datetime.date]] = None,
        has_labels: Optional[Dict[str, str]] = None,
        execution_states: Optional[Set[quantum.ExecutionStatus.State]] = None,
    ) -> List[AbstractJob]:
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
                 `quantum.ExecutionStatus.State` enum for accepted values.
        """
        valid_jobs: List[AbstractJob] = []
        for processor in self._processors.values():
            programs = processor.list_programs(
                created_before=created_before, created_after=created_after, has_labels=has_labels
            )
            for program in programs:
                valid_jobs.extend(
                    program.list_jobs(
                        created_before=created_before,
                        created_after=created_after,
                        has_labels=has_labels,
                        execution_states=execution_states,
                    )
                )
        return valid_jobs

    def list_processors(self) -> Sequence[AbstractLocalProcessor]:
        """Returns a list of Processors that the user has visibility to in the
        current Engine project. The names of these processors are used to
        identify devices when scheduling jobs and gathering calibration metrics.

        Returns:
            A list of EngineProcessors to access status, device and calibration
            information.
        """
        return list(self._processors.values())

    def get_processor(self, processor_id: str) -> AbstractLocalProcessor:
        """Returns an EngineProcessor for a Quantum Engine processor.

        Args:
            processor_id: The processor unique identifier.

        Returns:
            A EngineProcessor for the processor.
        """
        return self._processors[processor_id]

    def get_sampler(self, processor_id: Union[str, List[str]]) -> cirq.Sampler:
        """Returns a sampler backed by the engine.

        Args:
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.

        Raises:
            ValueError: if multiple processor ids are given.
        """
        if not isinstance(processor_id, str):
            raise ValueError(f'Invalid processor {processor_id}')
        return self._processors[processor_id].get_sampler()
