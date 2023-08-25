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

from typing import AbstractSet, Any, Optional, Tuple

import numpy as np

import cirq
from cirq._compat import proper_repr


@cirq.value_equality(approximate=True)
class CouplerPulse(cirq.ops.Gate):
    r"""Tunable pulse for entangling adjacent qubits.

    For experimental usage only.

    This operation sends a trapezoidal pulse to the coupler between two
    adjacent qubits placed in resonance.

    Note that this gate does not have a unitary matrix and must be
    characterized by the user in order to determine its effects.

    ```
                     __________
                    /          \
    _______________/            \________________
     |<---------->|  |<------>|  |<----------->|
      padding_time   hold_time     padding_time
                 ->  <-      ->   <-
                rise_time   rise_time
    ```

    Args:
            hold_time: Length of the 'plateau' part of the coupler trajectory.
            coupling_mhz: Target qubit-qubit coupling reached at the plateau.
            rise_time: Width of the rising (or falling) section of the trapezoidal pulse.
            padding_time: Symmetric padding around the coupler pulse.
            q0_detune_mhz: Detuning of the first qubit.
            q1_detune_mhz: Detuning of the second qubit.
    """

    def __init__(
        self,
        hold_time: cirq.Duration,
        coupling_mhz: cirq.TParamVal,
        rise_time: Optional[cirq.Duration] = cirq.Duration(nanos=8),
        padding_time: Optional[cirq.Duration] = cirq.Duration(nanos=2.5),
        q0_detune_mhz: cirq.TParamVal = 0.0,
        q1_detune_mhz: cirq.TParamVal = 0.0,
    ):
        """Inits CouplerPulse.

        Args:
            hold_time: Length of the 'plateau' part of the coupler trajectory.
            coupling_mhz: Target qubit-qubit coupling reached at the plateau.
            rise_time: Width of the rising (or falling) action of the trapezoidal pulse.
            padding_time: Symmetric padding around the coupler pulse.
            q0_detune_mhz: Detuning of the first qubit.
            q1_detune_mhz: Detuning of the second qubit.

        """
        self.hold_time = hold_time
        self.coupling_mhz = coupling_mhz
        self.rise_time = rise_time or cirq.Duration(nanos=8)
        self.padding_time = padding_time or cirq.Duration(nanos=2.5)
        self.q0_detune_mhz = q0_detune_mhz
        self.q1_detune_mhz = q1_detune_mhz

    def num_qubits(self) -> int:
        return 2

    def _unitary_(self) -> np.ndarray:
        return NotImplemented

    def __repr__(self) -> str:
        return (
            'cirq_google.experimental.ops.coupler_pulse.'
            f'CouplerPulse(hold_time={proper_repr(self.hold_time)}, '
            f'coupling_mhz={proper_repr(self.coupling_mhz)}, '
            f'rise_time={proper_repr(self.rise_time)}, '
            f'padding_time={proper_repr(self.padding_time)}, '
            f'q0_detune_mhz={proper_repr(self.q0_detune_mhz)}, '
            f'q1_detune_mhz={proper_repr(self.q1_detune_mhz)})'
        )

    def __str__(self) -> str:
        return (
            f'CouplerPulse(hold_time={self.hold_time}, '
            f'coupling_mhz={self.coupling_mhz}, '
            f'rise_time={self.rise_time}, '
            f'padding_time={self.padding_time}, '
            f'q0_detune_mhz={self.q0_detune_mhz}, '
            f'q1_detune_mhz={self.q1_detune_mhz})'
        )

    def _is_parameterized_(self) -> bool:
        return (
            cirq.is_parameterized(self.hold_time)
            or cirq.is_parameterized(self.coupling_mhz)
            or cirq.is_parameterized(self.rise_time)
            or cirq.is_parameterized(self.padding_time)
            or cirq.is_parameterized(self.q0_detune_mhz)
            or cirq.is_parameterized(self.q1_detune_mhz)
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        return (
            cirq.parameter_names(self.hold_time)
            | cirq.parameter_names(self.coupling_mhz)
            | cirq.parameter_names(self.rise_time)
            | cirq.parameter_names(self.padding_time)
            | cirq.parameter_names(self.q0_detune_mhz)
            | cirq.parameter_names(self.q1_detune_mhz)
        )

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolverOrSimilarType, recursive: bool
    ) -> 'CouplerPulse':
        return CouplerPulse(
            hold_time=cirq.resolve_parameters(self.hold_time, resolver, recursive=recursive),
            coupling_mhz=cirq.resolve_parameters(self.coupling_mhz, resolver, recursive=recursive),
            rise_time=cirq.resolve_parameters(self.rise_time, resolver, recursive=recursive),
            padding_time=cirq.resolve_parameters(self.padding_time, resolver, recursive=recursive),
            q0_detune_mhz=cirq.resolve_parameters(
                self.q0_detune_mhz, resolver, recursive=recursive
            ),
            q1_detune_mhz=cirq.resolve_parameters(
                self.q1_detune_mhz, resolver, recursive=recursive
            ),
        )

    def _value_equality_values_(self) -> Any:
        return (
            self.hold_time,
            self.coupling_mhz,
            self.rise_time,
            self.padding_time,
            self.q0_detune_mhz,
            self.q1_detune_mhz,
        )

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
        s = f'/‾‾({self.hold_time}@{self.coupling_mhz}MHz)‾‾\\'
        return (s, s)

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(
            self,
            [
                'hold_time',
                'coupling_mhz',
                'rise_time',
                'padding_time',
                'q0_detune_mhz',
                'q1_detune_mhz',
            ],
        )
