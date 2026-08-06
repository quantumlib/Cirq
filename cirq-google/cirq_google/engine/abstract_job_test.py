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
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import cirq
from cirq_google.engine.abstract_job import AbstractJob

if TYPE_CHECKING:
    import datetime

    import cirq_google.engine.abstract_engine as abstract_engine
    import cirq_google.engine.abstract_program as abstract_program


class MockProgram:
    def __init__(self, is_batch=True, keys=None):
        self._is_batch = is_batch
        self._keys = keys or []

    def is_batch(self) -> bool:
        return self._is_batch

    def batch_size(self) -> int:
        return len(self._keys)

    def batch_keys(self) -> list[str]:
        return self._keys


class MockJob(AbstractJob):
    def engine(self) -> abstract_engine.AbstractEngine:  # type: ignore[empty-body]
        pass

    def id(self) -> str:  # type: ignore[empty-body]
        pass

    def program(self) -> abstract_program.AbstractProgram:
        return getattr(self, '_mock_program', MockProgram())  # type: ignore[arg-type]

    def create_time(self) -> datetime.datetime:  # type: ignore[empty-body]
        pass

    def update_time(self) -> datetime.datetime:  # type: ignore[empty-body]
        pass

    def description(self) -> str:  # type: ignore[empty-body]
        pass

    def set_description(self, description: str) -> AbstractJob:  # type: ignore[empty-body]
        pass

    def labels(self) -> dict[str, str]:  # type: ignore[empty-body]
        pass

    def set_labels(self, labels: dict[str, str]) -> AbstractJob:  # type: ignore[empty-body]
        pass

    def add_labels(self, labels: dict[str, str]) -> AbstractJob:  # type: ignore[empty-body]
        pass

    def remove_labels(self, keys: list[str]) -> AbstractJob:  # type: ignore[empty-body]
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

    def get_config(self):
        pass

    def get_circuit(self, circuit_num: int | None = None) -> cirq.Circuit:
        return cirq.Circuit()

    def cancel(self) -> None:
        pass

    def delete(self) -> None:
        pass

    async def results_async(self):
        return [cirq.ResultDict(params={}, measurements={'a': np.asarray([t])}) for t in range(5)]

    async def batched_results_async(self):
        return [[r] for r in await self.results_async()]


def test_instantiation_and_iteration():
    job = MockJob()

    # Test length
    assert len(job) == 5

    # Test direct indexing
    assert job[3].measurements['a'][0] == 3

    #  Test iterating through for loop
    for count, result in enumerate(job):
        assert result.measurements['a'][0] == count

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


def test_get_circuit():
    job = MockJob()
    assert job.get_circuit() == cirq.Circuit()
    assert job.get_circuit(1) == cirq.Circuit()


def test_batched_results():
    job = MockJob()
    batched = job.batched_results()
    assert len(batched) == 5
    for count, r_list in enumerate(batched):
        assert len(r_list) == 1
        assert r_list[0].measurements['a'][0] == count


def test_mappings_results():
    job = MockJob()
    prog = MockProgram(is_batch=True, keys=['k0', 'k1', 'k2', 'k3', 'k4'])
    job._mock_program = prog
    assert prog.batch_size() == 5
    mapping = job.mappings_results()
    assert list(mapping.keys()) == ['k0', 'k1', 'k2', 'k3', 'k4']
    for count, (k, r_list) in enumerate(mapping.items()):
        assert len(r_list) == 1
        assert r_list[0].measurements['a'][0] == count


def test_mappings_results_key_length_mismatch():
    job = MockJob()
    job._mock_program = MockProgram(is_batch=True, keys=['k0', 'k1'])
    with pytest.raises(
        ValueError, match=r'Number of keys \(2\) does not match number of batch results \(5\)'
    ):
        _ = job.mappings_results()
