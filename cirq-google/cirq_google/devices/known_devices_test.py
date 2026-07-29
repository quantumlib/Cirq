# Copyright 2019 The Cirq Developers
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

import numpy as np
import pytest

import cirq
import cirq_google
import cirq_google.experimental.ops.coupler_pulse as coupler_pulse


def test_sycamore_grid_layout():
    # Qubits on Sycamore but not on Sycamore23
    q0 = cirq.GridQubit(5, 5)
    q1 = cirq.GridQubit(5, 6)
    syc = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1)
    sqrt_iswap = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)
    cirq_google.Sycamore.validate_operation(syc)
    cirq_google.Sycamore.validate_operation(sqrt_iswap)

    with pytest.raises(ValueError):
        cirq_google.Sycamore23.validate_operation(syc)
    with pytest.raises(ValueError):
        cirq_google.Sycamore23.validate_operation(sqrt_iswap)


@pytest.mark.parametrize(
    'device, qubit_size, layout_str',
    [
        (
            cirq_google.Sycamore,
            88,
            """\
                                             (0, 5)───(0, 6)
                                             │        │
                                             │        │
                                    (1, 4)───(1, 5)───(1, 6)───(1, 7)
                                    │        │        │        │
                                    │        │        │        │
                           (2, 3)───(2, 4)───(2, 5)───(2, 6)───(2, 7)───(2, 8)
                           │        │        │        │        │        │
                           │        │        │        │        │        │
                  (3, 2)───(3, 3)───(3, 4)───(3, 5)───(3, 6)───(3, 7)───(3, 8)───(3, 9)
                  │        │        │        │        │        │        │        │
                  │        │        │        │        │        │        │        │
         (4, 1)───(4, 2)───(4, 3)───(4, 4)───(4, 5)───(4, 6)───(4, 7)───(4, 8)───(4, 9)
         │        │        │        │        │        │        │        │
         │        │        │        │        │        │        │        │
(5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)───(5, 5)───(5, 6)───(5, 7)───(5, 8)
         │        │        │        │        │        │        │
         │        │        │        │        │        │        │
         (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)───(6, 6)───(6, 7)
                  │        │        │        │        │
                  │        │        │        │        │
                  (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
                           │        │        │
                           │        │        │
                           (8, 3)───(8, 4)───(8, 5)
                                    │
                                    │
                                    (9, 4)""",
        ),
        (
            cirq_google.Sycamore23,
            32,
            """\
                  (3, 2)
                  │
                  │
         (4, 1)───(4, 2)───(4, 3)
         │        │        │
         │        │        │
(5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)
         │        │        │        │
         │        │        │        │
         (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)
                  │        │        │        │
                  │        │        │        │
                  (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
                           │        │        │
                           │        │        │
                           (8, 3)───(8, 4)───(8, 5)
                                    │
                                    │
                                    (9, 4)""",
        ),
    ],
)
def test_sycamore_devices(device, qubit_size, layout_str):
    q0 = cirq.GridQubit(5, 3)
    q1 = cirq.GridQubit(5, 4)
    valid_sycamore_gates_and_ops = [
        cirq_google.SYC,
        cirq.SQRT_ISWAP,
        cirq.SQRT_ISWAP_INV,
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.Z(q0).with_tags(cirq_google.PhysicalZTag()),
        coupler_pulse.CouplerPulse(hold_time=cirq.Duration(nanos=10), coupling_mhz=25.0),
        cirq.measure(q0),
        cirq.WaitGate(cirq.Duration(millis=5)),
        # TODO(#5050) Uncomment after GlobalPhaseGate support is added.
        # cirq.GlobalPhaseGate(-1.0),
    ]
    syc = cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1)
    sqrt_iswap = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)

    assert str(device) == layout_str
    assert len(device.metadata.qubit_pairs) == qubit_size
    assert all(gate_or_op in device.metadata.gateset for gate_or_op in valid_sycamore_gates_and_ops)
    assert len(device.metadata.gate_durations) == len(device.metadata.gateset.gates)
    assert any(
        isinstance(cgs, cirq_google.SycamoreTargetGateset)
        for cgs in device.metadata.compilation_target_gatesets
    )
    assert any(
        isinstance(cgs, cirq.SqrtIswapTargetGateset)
        for cgs in device.metadata.compilation_target_gatesets
    )

    device.validate_operation(syc)
    device.validate_operation(sqrt_iswap)

    assert next(
        (
            duration
            for gate_family, duration in device.metadata.gate_durations.items()
            if syc in gate_family
        ),
        None,
    ) == cirq.Duration(nanos=12)
    assert next(
        (
            duration
            for gate_family, duration in device.metadata.gate_durations.items()
            if sqrt_iswap in gate_family
        ),
        None,
    ) == cirq.Duration(nanos=32)


@pytest.mark.parametrize(
    'device, qubit_size, layout_str',
    [
        (
            cirq_google.Willow,
            105,
            """\
                                                        (0, 6)────(0, 7)────(0, 8)
                                                        │         │         │
                                                        │         │         │
                                              (1, 5)────(1, 6)────(1, 7)────(1, 8)
                                              │         │         │         │
                                              │         │         │         │
                                    (2, 4)────(2, 5)────(2, 6)────(2, 7)────(2, 8)────(2, 9)────(2, 10)
                                    │         │         │         │         │         │         │
                                    │         │         │         │         │         │         │
                           (3, 3)───(3, 4)────(3, 5)────(3, 6)────(3, 7)────(3, 8)────(3, 9)────(3, 10)
                           │        │         │         │         │         │         │         │
                           │        │         │         │         │         │         │         │
                  (4, 2)───(4, 3)───(4, 4)────(4, 5)────(4, 6)────(4, 7)────(4, 8)────(4, 9)────(4, 10)────(4, 11)───(4, 12)
                  │        │        │         │         │         │         │         │         │          │         │
                  │        │        │         │         │         │         │         │         │          │         │
         (5, 1)───(5, 2)───(5, 3)───(5, 4)────(5, 5)────(5, 6)────(5, 7)────(5, 8)────(5, 9)────(5, 10)────(5, 11)───(5, 12)
         │        │        │        │         │         │         │         │         │         │          │         │
         │        │        │        │         │         │         │         │         │         │          │         │
(6, 0)───(6, 1)───(6, 2)───(6, 3)───(6, 4)────(6, 5)────(6, 6)────(6, 7)────(6, 8)────(6, 9)────(6, 10)────(6, 11)───(6, 12)───(6, 13)───(6, 14)
                  │        │        │         │         │         │         │         │         │          │         │         │
                  │        │        │         │         │         │         │         │         │          │         │         │
                  (7, 2)───(7, 3)───(7, 4)────(7, 5)────(7, 6)────(7, 7)────(7, 8)────(7, 9)────(7, 10)────(7, 11)───(7, 12)───(7, 13)
                  │        │        │         │         │         │         │         │         │          │         │
                  │        │        │         │         │         │         │         │         │          │         │
                  (8, 2)───(8, 3)───(8, 4)────(8, 5)────(8, 6)────(8, 7)────(8, 8)────(8, 9)────(8, 10)────(8, 11)───(8, 12)
                                    │         │         │         │         │         │         │          │
                                    │         │         │         │         │         │         │          │
                                    (9, 4)────(9, 5)────(9, 6)────(9, 7)────(9, 8)────(9, 9)────(9, 10)────(9, 11)
                                    │         │         │         │         │         │         │
                                    │         │         │         │         │         │         │
                                    (10, 4)───(10, 5)───(10, 6)───(10, 7)───(10, 8)───(10, 9)───(10, 10)
                                                        │         │         │         │
                                                        │         │         │         │
                                                        (11, 6)───(11, 7)───(11, 8)───(11, 9)
                                                        │         │         │
                                                        │         │         │
                                                        (12, 6)───(12, 7)───(12, 8)""",
        )
    ],
)
def test_willow_devices(device, qubit_size, layout_str):
    q0 = cirq.GridQubit(5, 3)
    q1 = cirq.GridQubit(5, 4)
    valid_willow_gates_and_ops = [
        cirq.CZ,
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.Z(q0).with_tags(cirq_google.PhysicalZTag()),
        cirq.measure(q0),
        cirq.WaitGate(cirq.Duration(millis=5)),
    ]
    cz = cirq.CZ(q0, q1)

    assert str(device) == layout_str
    assert len(device.metadata.qubit_pairs) == qubit_size
    assert all(gate_or_op in device.metadata.gateset for gate_or_op in valid_willow_gates_and_ops)

    device.validate_operation(cz)
