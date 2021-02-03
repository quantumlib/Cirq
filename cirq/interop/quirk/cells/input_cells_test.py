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

import pytest

from cirq import quirk_url_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.interop.quirk.cells.input_cells import SetDefaultInputCell


def test_missing_input_cell():
    with pytest.raises(ValueError, match='Missing input'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["+=A2"]]}')


def test_input_cell():
    assert_url_to_circuit_returns(
        '{"cols":[["inputA4",1,1,1,"+=A4"]]}',
        maps={
            0x_0_0: 0x_0_0,
            0x_2_3: 0x_2_5,
        },
    )

    assert_url_to_circuit_returns(
        '{"cols":[["inputA3",1,1,"inputB3",1,1,"+=AB3"]]}',
        maps={
            0o_0_0_0: 0o_0_0_0,
            0o_2_3_1: 0o_2_3_7,
            0o_1_1_0: 0o_1_1_1,
            0o_4_4_0: 0o_4_4_0,
        },
    )

    # Overlaps with effect.
    with pytest.raises(ValueError, match='Overlapping registers'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[["+=A3","inputA3"]]}'
        )


def test_reversed_input_cell():
    assert_url_to_circuit_returns(
        '{"cols":[["revinputA4",1,1,1,"+=A4"]]}',
        maps={
            0x_0_0: 0x_0_0,
            0x_2_3: 0x_2_7,
            0x_1_3: 0x_1_B,
        },
    )

    assert_url_to_circuit_returns(
        '{"cols":[["revinputA3",1,1,"revinputB3",1,1,"+=AB3"]]}',
        maps={
            0o_0_0_0: 0o_0_0_0,
            0o_2_6_1: 0o_2_6_7,
            0o_1_1_0: 0o_1_1_0,
            0o_4_4_0: 0o_4_4_1,
        },
    )

    # Overlaps with effect.
    with pytest.raises(ValueError, match='Overlapping registers'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[["+=A3","revinputA3"]]}'
        )


def test_set_default_input_cell():
    # Later column.
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":11}],["+=A4"]]}',
        maps={
            0: 11,
            4: 15,
            5: 0,
        },
    )

    # Same column.
    assert_url_to_circuit_returns(
        '{"cols":[["+=A4",{"id":"setA","arg":11}]]}',
        maps={
            0: 11,
            4: 15,
            5: 0,
        },
    )

    # Overwrite.
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":0}],["+=A4",{"id":"setA","arg":11}]]}',
        maps={
            0: 11,
            4: 15,
            5: 0,
        },
    )
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":11}],["+=A4",{"id":"setA","arg":0}]]}',
        maps={
            0: 0,
            4: 4,
            5: 5,
        },
    )

    # Different values over time.
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":1}],["+=A4"],[{"id":"setA","arg":4}],["+=A4"]]}',
        maps={
            0: 5,
        },
    )

    # Broadcast.
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":1}],["+=A2",1,"+=A2"],["+=A2",1,"+=A2"]]}',
        maps={
            0b_00_00: 0b_10_10,
            0b_10_01: 0b_00_11,
        },
    )

    # Too late.
    with pytest.raises(ValueError, match='Missing input'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[["+=A2"],[{"id":"setA","arg":1}]]}'
        )


def test_with_line_qubits_mapped_to():
    cell = SetDefaultInputCell('a', 5)
    assert cell.with_line_qubits_mapped_to([]) is cell
