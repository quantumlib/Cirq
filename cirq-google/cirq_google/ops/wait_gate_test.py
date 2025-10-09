# Copyright 2025 The Cirq Developers
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
import pytest
import sympy
import tunits as tu

import cirq
from cirq_google.ops import wait_gate as wg


def test_wait_gate_with_unit_init() -> None:
    g = wg.WaitGateWithUnit(1 * tu.us)
    assert g.duration == cirq.Duration(nanos=1000)

    g = wg.WaitGateWithUnit(1 * tu.us, num_qubits=2)
    assert g._qid_shape == (2, 2)

    g = wg.WaitGateWithUnit(sympy.Symbol("d"))
    assert g.duration == sympy.Symbol("d")

    with pytest.raises(ValueError, match="either be a tu.Value or a sympy.Symbol."):
        wg.WaitGateWithUnit(10)

    with pytest.raises(ValueError, match="Waiting on an empty set of qubits."):
        wg.WaitGateWithUnit(10 * tu.ns, qid_shape=())

    with pytest.raises(ValueError, match="num_qubits"):
        wg.WaitGateWithUnit(10 * tu.ns, qid_shape=(2, 2), num_qubits=5)


def test_wait_gate_with_units_resolving() -> None:
    gate = wg.WaitGateWithUnit(sympy.Symbol("d"))

    resolved_gate = cirq.resolve_parameters(gate, {"d": 10 * tu.ns})
    assert resolved_gate.duration == cirq.Duration(nanos=10)

    gate = wg.WaitGateWithUnit(10 * tu.ns)
    assert gate._resolve_parameters_(cirq.ParamResolver({}), True) == gate


def test_wait_gate_equality() -> None:
    gate1 = wg.WaitGateWithUnit(10 * tu.ns)
    gate2 = wg.WaitGateWithUnit(10 * tu.ns)
    assert gate1 == gate2

    gate_symbol_1 = wg.WaitGateWithUnit(sympy.Symbol("a"))
    gate_symbol_2 = wg.WaitGateWithUnit(sympy.Symbol("a"))
    assert gate_symbol_1 == gate_symbol_2
    assert gate_symbol_1 != gate1


def test_wait_gate_jsonify() -> None:
    gate = wg.WaitGateWithUnit(sympy.Symbol("d"))
    assert gate == cirq.read_json(json_text=cirq.to_json(gate))
