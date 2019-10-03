# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List, Iterator, Iterable

import pytest

import cirq
from cirq.contrib.quirk import quirk_url_to_circuit
from cirq.contrib.quirk.cells.cell import Cell, CELL_SIZES, CellMaker
from cirq.contrib.quirk.cells.input_cells import InputCell, SetDefaultInputCell
from cirq.contrib.quirk.cells.testing import assert_url_to_circuit_returns


def test_input_cell():
    # Overlaps.
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":11}],["+=A4"]]}',
        maps={
            0: 11,
            4: 15,
            5: 0,
        })

    # Overlaps with effect.
    with pytest.raises(ValueError, match='Overlapping operations'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["+=A3","inputA3"]]}')

    # Not present.
    with pytest.raises(ValueError, match='Missing input'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["+=A2"]]}')


def test_set_default_input_cell():
    # Later column.
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":11}],["+=A4"]]}',
        maps={
            0: 11,
            4: 15,
            5: 0,
        })

    # Same column.
    assert_url_to_circuit_returns('{"cols":[["+=A4",{"id":"setA","arg":11}]]}',
                                  maps={
                                      0: 11,
                                      4: 15,
                                      5: 0,
                                  })

    # Overwrite.
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":0}],["+=A4",{"id":"setA","arg":11}]]}',
        maps={
            0: 11,
            4: 15,
            5: 0,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":11}],["+=A4",{"id":"setA","arg":0}]]}',
        maps={
            0: 0,
            4: 4,
            5: 5,
        })

    # Different values over time.
    assert_url_to_circuit_returns(
        '{"cols":['
        '[{"id":"setA","arg":1}],'
        '["+=A4"],'
        '[{"id":"setA","arg":4}],'
        '["+=A4"]]}',
        maps={
            0: 5,
        })

    # Broadcast.
    assert_url_to_circuit_returns(
        '{"cols":['
        '[{"id":"setA","arg":1}],'
        '["+=A2",1,"+=A2"],'
        '["+=A2",1,"+=A2"]]}',
        maps={
            0b_00_00: 0b_10_10,
            0b_10_01: 0b_00_11,
        })

    # Too late.
    with pytest.raises(ValueError, match='Missing input'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["+=A2"],'
                                 '[{"id":"setA","arg":1}]]}')
