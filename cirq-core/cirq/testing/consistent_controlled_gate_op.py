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

from typing import Any, Sequence, Optional, Union, Collection

from cirq import protocols, devices, ops
import sympy


def assert_controlled_and_controlled_by_identical(
    gate: ops.Gate,
    *,
    exponents: Sequence[Any] = (0, 1, -1, 0.25, -0.5, 0.1, sympy.Symbol('s')),
    num_controls: Sequence[int] = (1, 2, 3, 10),
    control_values: Optional[Sequence[Optional[Sequence[Union[int, Collection[int]]]]]] = None,
):
    """Checks that gate.on().controlled_by() == gate.controlled().on()"""
    if control_values is not None:
        if len(num_controls) != len(control_values):
            raise ValueError(f"len(num_controls) != len(control_values)")
    for exponent in exponents:
        gate_exp = protocols.pow(gate, exponent, None)
        if not gate_exp:
            continue
        for i, num_control in enumerate(num_controls):
            control_value = control_values[i] if control_values else None
            if control_value is not None and len(control_value) != num_control:
                raise ValueError(f"len(control_values[{i}]) != num_controls[{i}]")
            _assert_gate_consistent(gate_exp, num_control, control_values)


def _assert_gate_consistent(
    gate: ops.Gate,
    num_controls: int,
    control_values: Optional[Sequence[Union[int, Collection[int]]]],
):
    qubits = devices.LineQubit.range(num_controls + protocols.num_qubits(gate))
    control_qubits = qubits[:num_controls]
    gate_qubits = qubits[num_controls:]
    gate_controlled_on = gate.controlled(num_controls, control_values).on(
        *control_qubits, *gate_qubits
    )
    gate_on_controlled_by = gate.on(*gate_qubits).controlled_by(
        *control_qubits, control_values=control_values
    )
    assert (
        gate_controlled_on == gate_on_controlled_by
    ), "gate.controlled().on() and gate.on().controlled() should return the same operations."
