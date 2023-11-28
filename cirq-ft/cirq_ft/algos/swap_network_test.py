# Copyright 2023 The Cirq Developers
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

import random

import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests

random.seed(12345)


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers",
    [[3, 5, 1], [2, 2, 3], [2, 3, 4], [3, 2, 5], [4, 1, 10]],
)
@allow_deprecated_cirq_ft_use_in_tests
def test_swap_with_zero_gate(selection_bitsize, target_bitsize, n_target_registers):
    # Construct the gate.
    gate = cirq_ft.SwapWithZeroGate(selection_bitsize, target_bitsize, n_target_registers)
    # Allocate selection and target qubits.
    all_qubits = cirq.LineQubit.range(cirq.num_qubits(gate))
    selection = all_qubits[:selection_bitsize]
    target = np.array(all_qubits[selection_bitsize:]).reshape((n_target_registers, target_bitsize))
    # Create a circuit.
    circuit = cirq.Circuit(gate.on_registers(selection=selection, target=target))

    # Load data[i] in i'th target register; where each register is of size target_bitsize
    data = [random.randint(0, 2**target_bitsize - 1) for _ in range(n_target_registers)]
    target_state = [int(x) for d in data for x in format(d, f"0{target_bitsize}b")]

    sim = cirq.Simulator(dtype=np.complex128)
    expected_state_vector = np.zeros(2**target_bitsize)
    # Iterate on every selection integer.
    for selection_integer in range(len(data)):
        # Load `selection_integer` in the selection register and construct initial state.
        selection_state = [int(x) for x in format(selection_integer, f"0{selection_bitsize}b")]
        initial_state = selection_state + target_state
        # Simulate the circuit with the initial state.
        result = sim.simulate(circuit, initial_state=initial_state)
        # Get the sub_state_vector corresponding to qubit register `target[0]`.
        result_state_vector = cirq.sub_state_vector(
            result.final_state_vector,
            keep_indices=list(range(selection_bitsize, selection_bitsize + target_bitsize)),
        )
        # Expected state vector should correspond to data[selection_integer] due to the swap.
        expected_state_vector[data[selection_integer]] = 1
        # Assert that result and expected state vectors are equal; reset and continue.
        assert cirq.equal_up_to_global_phase(result_state_vector, expected_state_vector)
        expected_state_vector[data[selection_integer]] = 0


@allow_deprecated_cirq_ft_use_in_tests
def test_swap_with_zero_gate_diagram():
    gate = cirq_ft.SwapWithZeroGate(3, 2, 4)
    q = cirq.LineQubit.range(cirq.num_qubits(gate))
    circuit = cirq.Circuit(gate.on_registers(**infra.split_qubits(gate.signature, q)))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ────@(r⇋0)───
       │
1: ────@(r⇋0)───
       │
2: ────@(r⇋0)───
       │
3: ────swap_0───
       │
4: ────swap_0───
       │
5: ────swap_1───
       │
6: ────swap_1───
       │
7: ────swap_2───
       │
8: ────swap_2───
       │
9: ────swap_3───
       │
10: ───swap_3───
""",
    )
    cirq.testing.assert_equivalent_repr(gate, setup_code='import cirq_ft')


@allow_deprecated_cirq_ft_use_in_tests
def test_multi_target_cswap():
    qubits = cirq.LineQubit.range(5)
    c, q_x, q_y = qubits[0], qubits[1:3], qubits[3:]
    cswap = cirq_ft.MultiTargetCSwap(2).on_registers(control=c, target_x=q_x, target_y=q_y)
    cswap_approx = cirq_ft.MultiTargetCSwapApprox(2).on_registers(
        control=c, target_x=q_x, target_y=q_y
    )
    setup_code = "import cirq\nimport cirq_ft"
    cirq.testing.assert_implements_consistent_protocols(cswap, setup_code=setup_code)
    cirq.testing.assert_implements_consistent_protocols(cswap_approx, setup_code=setup_code)
    circuit = cirq.Circuit(cswap, cswap_approx)
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───@──────@(approx)───
      │      │
1: ───×(x)───×(x)────────
      │      │
2: ───×(x)───×(x)────────
      │      │
3: ───×(y)───×(y)────────
      │      │
4: ───×(y)───×(y)────────
    """,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ---@--------@(approx)---
      |        |
1: ---swap_x---swap_x------
      |        |
2: ---swap_x---swap_x------
      |        |
3: ---swap_y---swap_y------
      |        |
4: ---swap_y---swap_y------
    """,
        use_unicode_characters=False,
    )


@allow_deprecated_cirq_ft_use_in_tests
def test_multi_target_cswap_make_on():
    qubits = cirq.LineQubit.range(5)
    c, q_x, q_y = qubits[:1], qubits[1:3], qubits[3:]
    cswap1 = cirq_ft.MultiTargetCSwap(2).on_registers(control=c, target_x=q_x, target_y=q_y)
    cswap2 = cirq_ft.MultiTargetCSwap.make_on(control=c, target_x=q_x, target_y=q_y)
    assert cswap1 == cswap2


@pytest.mark.skip(reason="Cirq-FT is deprecated, use Qualtran instead.")
def test_notebook():
    execute_notebook('swap_network')


@pytest.mark.parametrize("n", [*range(1, 6)])
@allow_deprecated_cirq_ft_use_in_tests
def test_t_complexity(n):
    g = cirq_ft.MultiTargetCSwap(n)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)

    g = cirq_ft.MultiTargetCSwapApprox(n)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(g)


@pytest.mark.parametrize(
    "selection_bitsize, target_bitsize, n_target_registers, want",
    [
        [3, 5, 1, (0, 0)],
        [2, 2, 3, (16, 86)],
        [2, 3, 4, (36, 195)],
        [3, 2, 5, (32, 172)],
        [4, 1, 10, (36, 189)],
    ],
)
@allow_deprecated_cirq_ft_use_in_tests
def test_swap_with_zero_t_complexity(selection_bitsize, target_bitsize, n_target_registers, want):
    t_complexity = cirq_ft.TComplexity(t=want[0], clifford=want[1])
    gate = cirq_ft.SwapWithZeroGate(selection_bitsize, target_bitsize, n_target_registers)
    assert t_complexity == cirq_ft.t_complexity(gate)
