# Copyright 2020 The Cirq Developers
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

from typing import Any, Optional, Tuple

import numpy as np

import cirq


_MIN_DURATION = cirq.Duration(nanos=0)
_MAX_DURATION = cirq.Duration(nanos=100)


@cirq.value_equality(approximate=True)
class CouplerPulse(cirq.ops.gate_features.TwoQubitGate):
    """Tunable pulse for entangling adjacent qubits.

    For experimental usage only.

    This operation sends a trapezoidal pulse to the coupler between two
    adjacent qubits placed in resonance.

    Note that this gate does not have a unitary matrix and must be
    characterized by the user in order to determine its effects.

    Args:
            hold_time: Length of the 'plateau' part of the coupler trajectory.
            coupling_MHz: Target qubit-qubit coupling reached at the plateau.
            rise_time: Full width of the smoothstep rise/fall.
            padding_time: Symmetric padding around the coupler pulse.
    """

    def __init__(
        self,
        hold_time: cirq.Duration,
        coupling_mhz: float,
        rise_time: Optional[cirq.Duration] = cirq.Duration(nanos=8),
        padding_time: Optional[cirq.Duration] = cirq.Duration(nanos=2.5),
    ):
        """
        Args:
            hold_time: Length of the 'plateau' part of the coupler trajectory.
            coupling_MHz: Target qubit-qubit coupling reached at the plateau.
            rise_time: Full width of the smoothstep rise/fall.
            padding_time: Symmetric padding around the coupler pulse.
        """
        # Verification
        if hold_time > rise_time:
            raise ValueError(
                f'Full rise time {rise_time} must be longer '
                'than hold_time {hold_time} for CouplerPulse'
            )
        if hold_time < _MIN_DURATION or hold_time > _MAX_DURATION:
            raise ValueError(f'hold_time must be between {_MIN_DURATION} and {_MAX_DURATION}')
        if padding_time < _MIN_DURATION or padding_time > _MAX_DURATION:
            raise ValueError(f'padding_time must be between {_MIN_DURATION} and {_MAX_DURATION}')
        if rise_time < _MIN_DURATION or rise_time > _MAX_DURATION:
            raise ValueError(f'rise_time must be between {_MIN_DURATION} and {_MAX_DURATION}')

        self.hold_time = hold_time
        self.coupling_mhz = coupling_mhz
        self.rise_time = rise_time or cirq.Duration(nanos=8)
        self.padding_time = padding_time or cirq.Duration(nanos=2.5)

    def num_qubits(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return NotImplemented

    def __repr__(self) -> str:
        return (
            'cirq_google.experimental.ops.coupler_pulse.'
            + f'CouplerPulse(hold_time={self.hold_time!r}, '
            + f'coupling_mhz={self.coupling_mhz}, '
            + f'rise_time={self.rise_time!r}, '
            + f'padding_time={self.padding_time!r})'
        )

    def __str__(self) -> str:
        return (
            f'CouplerPulse(hold_time={self.hold_time}, '
            + f'coupling_mhz={self.coupling_mhz}, '
            + f'rise_time={self.rise_time}, '
            + f'padding_time={self.padding_time})'
        )

    def _value_equality_values_(self) -> Any:
        return self.hold_time, self.coupling_mhz, self.rise_time, self.padding_time

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
        s = f'/‾‾({self.hold_time}@{self.coupling_mhz}MHz)‾‾\\'
        return (s, s)

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(
            self, ['hold_time', 'coupling_mhz', 'rise_time', 'padding_time']
        )
