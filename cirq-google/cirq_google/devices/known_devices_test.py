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
