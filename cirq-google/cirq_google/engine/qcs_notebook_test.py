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

import unittest.mock as mock
import pytest

import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook, QCSObjectsForNotebook


def _assert_correct_types(result: QCSObjectsForNotebook):
    assert isinstance(result.device, cg.GridDevice)
    assert isinstance(result.sampler, cg.ProcessorSampler)
    assert isinstance(result.engine, cg.engine.AbstractEngine)
    assert isinstance(result.processor, cg.engine.AbstractProcessor)


def _assert_simulated_values(result: QCSObjectsForNotebook):
    assert not result.signed_in
    assert result.is_simulator
    assert result.project_id == 'fake_project'


def test_get_qcs_objects_for_notebook_virtual():
    result = get_qcs_objects_for_notebook(virtual=True)
    _assert_correct_types(result)
    _assert_simulated_values(result)
    assert result.processor_id == 'rainbow'
    assert len(result.device.metadata.qubit_set) == 23

    result = get_qcs_objects_for_notebook(processor_id='weber', virtual=True)
    _assert_correct_types(result)
    _assert_simulated_values(result)
    assert result.processor_id == 'weber'
    assert len(result.device.metadata.qubit_set) == 53


@mock.patch('cirq_google.engine.qcs_notebook.get_engine')
def test_get_qcs_objects_for_notebook_mocked_engine_fails(engine_mock):
    """Tests creating an engine object which fails."""
    engine_mock.side_effect = EnvironmentError('This is a mock, not real credentials.')
    result = get_qcs_objects_for_notebook()
    _assert_correct_types(result)
    _assert_simulated_values(result)


@mock.patch('cirq_google.engine.qcs_notebook.get_engine')
def test_get_qcs_objects_for_notebook_mocked_engine_succeeds(engine_mock):
    """Uses a mocked engine call to test a 'prod' Engine."""
    fake_processor = cg.engine.SimulatedLocalProcessor(
        processor_id='tester', project_name='mock_project', device=cg.Sycamore
    )
    fake_processor2 = cg.engine.SimulatedLocalProcessor(
        processor_id='tester23', project_name='mock_project', device=cg.Sycamore23
    )
    fake_engine = cg.engine.SimulatedLocalEngine([fake_processor, fake_processor2])
    engine_mock.return_value = fake_engine

    result = get_qcs_objects_for_notebook()
    _assert_correct_types(result)
    assert result.signed_in
    assert not result.is_simulator
    assert result.project_id == 'mock_project'
    assert len(result.device.metadata.qubit_set) == 54

    result = get_qcs_objects_for_notebook(processor_id='tester')
    _assert_correct_types(result)
    assert result.signed_in
    assert not result.is_simulator
    assert result.project_id == 'mock_project'
    assert len(result.device.metadata.qubit_set) == 54

    result = get_qcs_objects_for_notebook(processor_id='tester23')
    _assert_correct_types(result)
    assert result.signed_in
    assert not result.is_simulator
    assert result.project_id == 'mock_project'
    assert len(result.device.metadata.qubit_set) == 23


@mock.patch('cirq_google.engine.qcs_notebook.get_engine')
def test_get_qcs_objects_for_notebook_no_processors(engine_mock):
    fake_engine = cg.engine.SimulatedLocalEngine([])
    engine_mock.return_value = fake_engine
    with pytest.raises(ValueError, match='processors'):
        _ = get_qcs_objects_for_notebook()
