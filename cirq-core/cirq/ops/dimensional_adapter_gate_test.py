# Copyright 2018 The Cirq Developers
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

import numpy as np
import pytest
import sympy

import cirq


@pytest.mark.parametrize('split', [True, False])
def test_simulate_qudits_slices(split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    simulator = cirq.Simulator(split_untangled_states=split)

    circuit = cirq.Circuit(
        cirq.DimensionAdapterGate(cirq.X, [(3, (0, 1))])(q0),
        cirq.DimensionAdapterGate(cirq.X, [(4, (0, 3))])(q1),
    )
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    expected = np.zeros(12)
    expected[4 * 1 + 3] = 1

    np.testing.assert_almost_equal(result.final_state_vector, expected)
    assert len(result.measurements) == 0

    cirq.testing.assert_has_diagram(
        circuit,
        """
0 (d=3): ───X(subspace [0, 1])───

1 (d=4): ───X(subspace [0, 3])───
""",
        use_unicode_characters=True,
    )


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable(resolve_fn):
    a = sympy.Symbol('a')
    cy = cirq.DimensionAdapterGate(cirq.Y, [(3, slice(0, 2, 1))])
    cya = cirq.DimensionAdapterGate(cirq.YPowGate(exponent=a), [(3, slice(0, 2, 1))])
    assert cirq.is_parameterized(cya)
    assert not cirq.is_parameterized(cy)
    assert resolve_fn(cya, cirq.ParamResolver({'a': 1})) == cy


@pytest.mark.parametrize(
    'gate',
    [
        cirq.X,
        cirq.X ** 0.5,
        cirq.rx(np.pi),
        cirq.rx(np.pi / 2),
        cirq.Z,
        cirq.H,
    ],
)
def test_controlled_gate_is_consistent(gate: cirq.Gate):
    cgate = cirq.DimensionAdapterGate(cirq.X, [(3, (0, 2))])
    cirq.testing.assert_implements_consistent_protocols(cgate)
