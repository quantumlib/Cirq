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
from typing import Dict, List, TYPE_CHECKING
import pytest
import numpy as np
import cirq
from cirq_google.engine.abstract_job import AbstractJob

if TYPE_CHECKING:
    import datetime
    import cirq_google.engine.abstract_engine as abstract_engine
    import cirq_google.engine.abstract_processor as abstract_processor
    import cirq_google.engine.abstract_program as abstract_program


class MockJob(AbstractJob):
    def engine(self) -> 'abstract_engine.AbstractEngine':
        pass

    def id(self) -> str:
        pass

    def program(self) -> 'abstract_program.AbstractProgram':
        pass

    def create_time(self) -> 'datetime.datetime':
        pass

    def update_time(self) -> 'datetime.datetime':
        pass

    def description(self) -> str:
        pass

    def set_description(self, description: str) -> 'AbstractJob':
        pass

    def labels(self) -> Dict[str, str]:
        pass

    def set_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        pass

    def add_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        pass

    def remove_labels(self, keys: List[str]) -> 'AbstractJob':
        pass

    def processor_ids(self):
        pass

    def execution_status(self):
        pass

    def failure(self):
        pass

    def get_repetitions_and_sweeps(self):
        pass

    def get_processor(self):
        pass

    def get_calibration(self):
        pass

    def cancel(self) -> None:
        pass

    def delete(self) -> None:
        pass

    async def batched_results_async(self):
        pass

    async def results_async(self):
        return [cirq.ResultDict(params={}, measurements={'a': np.asarray([t])}) for t in range(5)]

    async def calibration_results_async(self):
        pass


def test_instantiation_and_iteration():
    job = MockJob()

    # Test length
    assert len(job) == 5

    # Test direct indexing
    assert job[3].measurements['a'][0] == 3

    #  Test iterating through for loop
    count = 0
    for result in job:
        assert result.measurements['a'][0] == count
        count += 1

    # Test iterator using iterator
    iterator = iter(job)
    result = next(iterator)
    assert result.measurements['a'][0] == 0
    result = next(iterator)
    assert result.measurements['a'][0] == 1
    result = next(iterator)
    assert result.measurements['a'][0] == 2
    result = next(iterator)
    assert result.measurements['a'][0] == 3
    result = next(iterator)
    assert result.measurements['a'][0] == 4
    with pytest.raises(StopIteration):
        next(iterator)
