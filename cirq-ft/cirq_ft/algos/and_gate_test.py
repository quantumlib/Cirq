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

import itertools
import random
from typing import List, Tuple

import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook

random.seed(12345)


@pytest.mark.parametrize("cv", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_and_gate(cv: Tuple[int, int]):
    c1, c2, t = cirq.LineQubit.range(3)
    input_states = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]
    output_states = [inp[:2] + (1 if inp[:2] == cv else 0,) for inp in input_states]

    and_gate = cirq_ft.And(cv)
    circuit = cirq.Circuit(and_gate.on(c1, c2, t))
    for inp, out in zip(input_states, output_states):
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, [c1, c2, t], inp, out)


def random_cv(n: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(n)]


@pytest.mark.parametrize("cv", [[1] * 3, random_cv(5), random_cv(6), random_cv(7)])
def test_multi_controlled_and_gate(cv: List[int]):
    gate = cirq_ft.And(cv)
    r = gate.signature
    assert r.get_right('junk').total_bits() == r.get_left('ctrl').total_bits() - 2
    quregs = infra.get_named_qubits(r)
    and_op = gate.on_registers(**quregs)
    circuit = cirq.Circuit(and_op)

    input_controls = [cv] + [random_cv(len(cv)) for _ in range(10)]
    qubit_order = infra.merge_qubits(gate.signature, **quregs)

    for input_control in input_controls:
        initial_state = input_control + [0] * (r.get_right('junk').total_bits() + 1)
        result = cirq.Simulator(dtype=np.complex128).simulate(
            circuit, initial_state=initial_state, qubit_order=qubit_order
        )
        expected_output = np.asarray([0, 1] if input_control == cv else [1, 0])
        assert cirq.equal_up_to_global_phase(
            cirq.sub_state_vector(
                result.final_state_vector, keep_indices=[cirq.num_qubits(gate) - 1]
            ),
            expected_output,
        )

        # Test adjoint.
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(
            circuit + cirq.Circuit(cirq.inverse(and_op)),
            qubit_order=qubit_order,
            inputs=initial_state,
            outputs=initial_state,
        )


def test_and_gate_diagram():
    gate = cirq_ft.And((1, 0, 1, 0, 1, 0))
    qubit_regs = infra.get_named_qubits(gate.signature)
    op = gate.on_registers(**qubit_regs)
    ctrl, junk, target = (
        qubit_regs["ctrl"].flatten(),
        qubit_regs["junk"].flatten(),
        qubit_regs['target'].flatten(),
    )
    # Qubit order should be alternating (control, ancilla) pairs.
    c_and_a = sum(zip(ctrl[1:], junk), ()) + (ctrl[-1],)
    qubit_order = np.concatenate([ctrl[0:1], c_and_a, target])
    # Test diagrams.
    cirq.testing.assert_has_diagram(
        cirq.Circuit(op),
        """
ctrl[0]: ───@─────
            │
ctrl[1]: ───(0)───
            │
junk[0]: ───Anc───
            │
ctrl[2]: ───@─────
            │
junk[1]: ───Anc───
            │
ctrl[3]: ───(0)───
            │
junk[2]: ───Anc───
            │
ctrl[4]: ───@─────
            │
junk[3]: ───Anc───
            │
ctrl[5]: ───(0)───
            │
target: ────And───
""",
        qubit_order=qubit_order,
    )
    cirq.testing.assert_has_diagram(
        cirq.Circuit(op**-1),
        """
ctrl[0]: ───@──────
            │
ctrl[1]: ───(0)────
            │
junk[0]: ───Anc────
            │
ctrl[2]: ───@──────
            │
junk[1]: ───Anc────
            │
ctrl[3]: ───(0)────
            │
junk[2]: ───Anc────
            │
ctrl[4]: ───@──────
            │
junk[3]: ───Anc────
            │
ctrl[5]: ───(0)────
            │
target: ────And†───
""",
        qubit_order=qubit_order,
    )
    # Test diagram of decomposed 3-qubit and ladder.
    decomposed_circuit = cirq.Circuit(cirq.decompose_once(op)) + cirq.Circuit(
        cirq.decompose_once(op**-1)
    )
    cirq.testing.assert_has_diagram(
        decomposed_circuit,
        """
ctrl[0]: ───@─────────────────────────────────────────────────────────@──────
            │                                                         │
ctrl[1]: ───(0)───────────────────────────────────────────────────────(0)────
            │                                                         │
junk[0]: ───And───@────────────────────────────────────────────@──────And†───
                  │                                            │
ctrl[2]: ─────────@────────────────────────────────────────────@─────────────
                  │                                            │
junk[1]: ─────────And───@───────────────────────────────@──────And†──────────
                        │                               │
ctrl[3]: ───────────────(0)─────────────────────────────(0)──────────────────
                        │                               │
junk[2]: ───────────────And───@──────────────────@──────And†─────────────────
                              │                  │
ctrl[4]: ─────────────────────@──────────────────@───────────────────────────
                              │                  │
junk[3]: ─────────────────────And───@─────@──────And†────────────────────────
                                    │     │
ctrl[5]: ───────────────────────────(0)───(0)────────────────────────────────
                                    │     │
target: ────────────────────────────And───And†───────────────────────────────
""",
        qubit_order=qubit_order,
    )


@pytest.mark.parametrize(
    "cv, adjoint, str_output",
    [
        ((1, 1, 1), False, "And"),
        ((1, 0, 1), False, "And(1, 0, 1)"),
        ((1, 1, 1), True, "And†"),
        ((1, 0, 1), True, "And†(1, 0, 1)"),
    ],
)
def test_and_gate_str_and_repr(cv, adjoint, str_output):
    gate = cirq_ft.And(cv, adjoint=adjoint)
    assert str(gate) == str_output
    cirq.testing.assert_equivalent_repr(gate, setup_code="import cirq_ft\n")


@pytest.mark.parametrize("cv", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_and_gate_adjoint(cv: Tuple[int, int]):
    c1, c2, t = cirq.LineQubit.range(3)
    all_cvs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    input_states = [inp + (1 if inp == cv else 0,) for inp in all_cvs]
    output_states = [inp + (0,) for inp in all_cvs]

    circuit = cirq.Circuit(cirq_ft.And(cv, adjoint=True).on(c1, c2, t))
    for inp, out in zip(input_states, output_states):
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, [c1, c2, t], inp, out)


def test_notebook():
    execute_notebook('and_gate')


@pytest.mark.parametrize(
    "cv", [*itertools.chain(*[itertools.product(range(2), repeat=n) for n in range(2, 7 + 1)])]
)
@pytest.mark.parametrize("adjoint", [*range(2)])
def test_t_complexity(cv, adjoint):
    gate = cirq_ft.And(cv=cv, adjoint=adjoint)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(gate)


def test_and_gate_raises():
    with pytest.raises(ValueError, match="at-least 2 control values"):
        _ = cirq_ft.And(cv=(1,))


def test_and_gate_power():
    cv = (1, 0)
    and_gate = cirq_ft.And(cv)
    assert and_gate**1 is and_gate
    assert and_gate**-1 == cirq_ft.And(cv, adjoint=True)
    assert (and_gate**-1) ** -1 == cirq_ft.And(cv)
