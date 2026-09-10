# Copyright 2026 The Cirq Developers
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

"""Multi-step multi-level reset gate.

This module defines `MultiStepMultiLevelReset`, an active reset gate for
superconducting qubits that resets the qubit to its ground state by iteratively
swapping excited-state populations into a fast-decaying reset resonator across
multiple frequency steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import attrs
import cirq

if TYPE_CHECKING:
    from cirq_google.study import symbol_util as su


@attrs.frozen(eq=False, hash=False)
class MultiStepMultiLevelReset(cirq.Gate):
    """Multi-step multi-level reset gate.

    This gate actively resets a superconducting qubit to the ground state by
    iteratively swapping excited populations (including higher levels like |2>
    and |1>) into an attached reset resonator across multiple frequency steps.
    Each step specifies a duration (`lengths`), a swap frequency detuning relative
    to the reset resonator frequency (`f_swaps_delta`), and a coupling strength
    (`gs`).

    Parameters that are not specified (set to `None`) are populated from calibration
    or device defaults at scheduling/runtime.

    Attributes:
        f_start: Starting frequency of the reset trajectory.
        already_at_readout_detuning: If True, indicates that the qubit is already
            at the readout detuning frequency prior to this gate (e.g. immediately
            following readout), rather than at idle. This is used to compute the
            step pulse amplitude needed to transition into `f_start`.
        f_end: Ending frequency of the reset trajectory.
        end_at_idle: If True, the qubit returns to idle frequency at the end of
            the gate, ignoring `f_end`. If False, the trajectory ends at `f_end`
            and an additional step pulse is applied to return to idle.
        lengths: Sequence of durations for each reset step.
        f_swaps_delta: Sequence of swap detuning frequencies for each step,
            measured relative to the reset resonator frequency.
        gs: Sequence of coupling strengths for each reset step.
        padding_before: Padding duration before the reset trajectory begins.
        padding_after: Padding duration after the reset trajectory ends.
        detune_to_start_freq: If True, include a detune step to bring the qubit
            to `f_start`. If False, assume the qubit is already at `f_start`.
        start_at_readout_detuning: If True, begins the reset trajectory at the
            qubit's readout detuning frequency, overriding `f_start`. This differs
            from `already_at_readout_detuning`, which specifies the qubit's initial
            state before the gate starts.
        coupler_amplitudes: Optional map of coupler names to detune amplitudes
            for couplers connected to this qubit during reset.
        compensate_coupled_qubit: If True, compensates for coupler detuning by
            applying a compensation detune to the other qubit connected to each
            coupler.
    """

    f_start: su.ValueOrSymbol | None = None
    already_at_readout_detuning: bool | None = None
    f_end: su.ValueOrSymbol | None = None
    end_at_idle: bool | None = None
    lengths: Sequence[su.ValueOrSymbol] | None = None
    f_swaps_delta: Sequence[su.ValueOrSymbol] | None = None
    gs: Sequence[su.ValueOrSymbol] | None = None
    padding_before: su.ValueOrSymbol | None = None
    padding_after: su.ValueOrSymbol | None = None
    detune_to_start_freq: bool | None = None
    start_at_readout_detuning: bool | None = None
    coupler_amplitudes: dict[str, cirq.TParamVal] | None = None
    compensate_coupled_qubit: bool | None = None

    def _num_qubits_(self) -> int:
        return 1

    def is_reset_gate(self) -> bool:
        return True

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> list[str]:
        return ["[R (MSML)]"]

    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> list[cirq.Operation]:
        return list(cirq.reset_each(*qubits))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MultiStepMultiLevelReset):
            return NotImplemented
        return (
            self.f_start == other.f_start
            and self.already_at_readout_detuning == other.already_at_readout_detuning
            and self.f_end == other.f_end
            and self.end_at_idle == other.end_at_idle
            and (tuple(self.lengths) if self.lengths is not None else None)
            == (tuple(other.lengths) if other.lengths is not None else None)
            and (tuple(self.f_swaps_delta) if self.f_swaps_delta is not None else None)
            == (tuple(other.f_swaps_delta) if other.f_swaps_delta is not None else None)
            and (tuple(self.gs) if self.gs is not None else None)
            == (tuple(other.gs) if other.gs is not None else None)
            and self.padding_before == other.padding_before
            and self.padding_after == other.padding_after
            and self.detune_to_start_freq == other.detune_to_start_freq
            and self.start_at_readout_detuning == other.start_at_readout_detuning
            and self.coupler_amplitudes == other.coupler_amplitudes
            and self.compensate_coupled_qubit == other.compensate_coupled_qubit
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.f_start,
                self.already_at_readout_detuning,
                self.f_end,
                self.end_at_idle,
                tuple(self.lengths) if self.lengths is not None else None,
                tuple(self.f_swaps_delta) if self.f_swaps_delta is not None else None,
                tuple(self.gs) if self.gs is not None else None,
                self.padding_before,
                self.padding_after,
                self.detune_to_start_freq,
                self.start_at_readout_detuning,
                (
                    tuple(sorted(self.coupler_amplitudes.items()))
                    if self.coupler_amplitudes is not None
                    else None
                ),
                self.compensate_coupled_qubit,
            )
        )

    def __repr__(self) -> str:
        args = []
        for field in attrs.fields(type(self)):
            val = getattr(self, field.name)
            if val is not None:
                args.append(f'{field.name}={val!r}')
        return f"cirq_google.MultiStepMultiLevelReset({', '.join(args)})"

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(
            self,
            [
                'f_start',
                'already_at_readout_detuning',
                'f_end',
                'end_at_idle',
                'lengths',
                'f_swaps_delta',
                'gs',
                'padding_before',
                'padding_after',
                'detune_to_start_freq',
                'start_at_readout_detuning',
                'coupler_amplitudes',
                'compensate_coupled_qubit',
            ],
        )
