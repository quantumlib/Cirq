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

"""A Multi-Moment Gauge Transformer for the cphase gate."""

from __future__ import annotations

from typing import cast

import numpy as np

from cirq import circuits, ops
from cirq.transformers.gauge_compiling.multi_moment_gauge_compiling import (
    MultiMomentGaugeTransformer,
)


class _PauliAndZPow:
    """In pulling through, one qubit gate can be represented by a Pauli and an Rz gate.
    The order is --Pauli--ZPowGate--.
    """

    pauli: ops.Pauli | ops.IdentityGate = ops.I
    zpow: ops.ZPowGate = ops.ZPowGate(exponent=0)

    commuting_gates = {ops.I, ops.Z}  # I,Z Commute with ZPowGate and CZPowGate; X,Y anti-commute.

    def __init__(
        self,
        pauli: ops.Pauli | ops.IdentityGate = ops.I,
        zpow: ops.ZPowGate = ops.ZPowGate(exponent=0),
    ) -> None:
        self.pauli = pauli
        self.zpow = zpow

    def _merge_left_zpow(self, left: ops.ZPowGate):
        """Merges ZPowGate from left."""
        if self.pauli in self.commuting_gates:
            self.zpow = ops.ZPowGate(exponent=left.exponent + self.zpow.exponent)
        else:
            self.zpow = ops.ZPowGate(exponent=-left.exponent + self.zpow.exponent)

    def _merge_right_zpow(self, right: ops.ZPowGate):
        """Merges ZPowGate from right."""
        self.zpow = ops.ZPowGate(exponent=right.exponent + self.zpow.exponent)

    def _merge_left_pauli(self, left: ops.Pauli):
        """Merges --left_pauli--self--."""
        if self.pauli == ops.I:
            self.pauli = left
        else:
            self.pauli = left.phased_pauli_product(self.pauli)[1]

    def _merge_right_pauli(self, right: ops.Pauli):
        """Merges --self--right_pauli--."""
        if self.pauli == ops.I:
            self.pauli = right
        else:
            self.pauli = right.phased_pauli_product(self.pauli)[1]
        if right not in self.commuting_gates:
            self.zpow = ops.ZPowGate(exponent=-self.zpow.exponent)

    def merge_left(self, left: _PauliAndZPow) -> None:
        """Inplace merge other from left."""
        self._merge_left_zpow(left.zpow)
        if left.pauli != ops.I:
            self._merge_left_pauli(cast(ops.Pauli, left.pauli))

    def merge_right(self, right: _PauliAndZPow) -> None:
        """Inplace merge other from right."""
        if right.pauli != ops.I:
            self._merge_right_pauli(cast(ops.Pauli, right.pauli))
        self._merge_right_zpow(right.zpow)

    def after_cphase(
        self, cphase: ops.CZPowGate
    ) -> tuple[ops.CZPowGate, _PauliAndZPow, _PauliAndZPow]:
        """Pull self through cphase.

        Returns:
            A tuple of
                (updated cphase gate, pull_through of this qubit, pull_through of the other qubit).
        """
        if self.pauli in self.commuting_gates:
            return cphase, self, _PauliAndZPow()
        else:
            # Taking self.pauli==X gate as an example:
            # 0: ─X─Z^t──@──────      0: ─X──@─────Z^t─       0: ─@──────X──Z^t──
            #            │       ==>         │           ==>      │
            # 1: ────────@^exp──      1: ────@^exp─────       1: ─@^-exp─Z^exp───
            # Similarly for X|Y on qubit 0/1, the result is always flipping cphase and
            # add an extra Rz rotation on the other qubit.
            return (
                cast(ops.CZPowGate, cphase**-1),
                self,
                _PauliAndZPow(zpow=ops.ZPowGate(exponent=cphase.exponent)),
            )

    def after_pauli(self, pauli: ops.Pauli | ops.IdentityGate) -> _PauliAndZPow:
        """Calculates ─self─pauli─  ==>  ─pauli─output─."""
        if pauli in self.commuting_gates:
            return _PauliAndZPow(self.pauli, self.zpow)
        else:
            return _PauliAndZPow(self.pauli, ops.ZPowGate(exponent=-self.zpow.exponent))

    def after_zpow(self, zpow: ops.ZPowGate) -> tuple[ops.ZPowGate, _PauliAndZPow]:
        """Calculates ─self─zpow─  ==>  ─zpow'─output─."""
        if self.pauli in self.commuting_gates:
            return zpow, self
        else:
            return ops.ZPowGate(exponent=-zpow.exponent), self

    def __str__(self) -> str:
        return f"─{self.pauli}──{self.zpow}─"

    def to_single_qubit_gate(self) -> ops.PhasedXZGate | ops.ZPowGate | ops.IdentityGate:
        """Converts the _PhasedXYAndRz to a single-qubit gate."""
        exp = self.zpow.exponent
        match self.pauli:
            case ops.I:
                if exp % 2 == 0:
                    return ops.I
                return self.zpow
            case ops.X:
                return ops.PhasedXZGate(x_exponent=1, z_exponent=exp, axis_phase_exponent=0)
            case ops.Y:
                return ops.PhasedXZGate(x_exponent=1, z_exponent=exp - 1, axis_phase_exponent=0)
            case _:  # ops.Z
                if (exp + 1) % 2 == 0:
                    return ops.I
                return ops.ZPowGate(exponent=1 + exp)


def _pull_through_single_cphase(
    cphase: ops.CZPowGate, input0: _PauliAndZPow, input1: _PauliAndZPow
) -> tuple[ops.CZPowGate, _PauliAndZPow, _PauliAndZPow]:
    """Pulls input0 and input1 through a CZPowGate.
    Input:
    0: ─(input0)─@─────
                 │
    1: ─(input1)─@^exp─
    Output:
    0: ─@────────(output0)─
        │
    1: ─@^+/-exp─(output1)─
    """

    # Step 1; pull input0 through CZPowGate.
    # 0: ─input0─@─────     0: ────────@─────────output0─
    #            │      ==>            │
    # 1: ─input1─@^exp─     1: ─input1─@^+/-exp──output1─
    output_cphase, output0, output1 = input0.after_cphase(cphase)

    # Step 2; similar to step 1, pull input1 through CZPowGate.
    #      0: ─@──────────pulled0────output0─     0: ─@────────output0─
    #  ==>     │                              ==>     │
    #      1: ─@^+/-exp───pulled1────output1─     1: ─@^+/-exp─output1─
    output_cphase, pulled1, pulled0 = input1.after_cphase(output_cphase)
    output0.merge_left(pulled0)
    output1.merge_left(pulled1)

    return output_cphase, output0, output1


_TARGET_GATESET: ops.Gateset = ops.Gateset(ops.CZPowGate)
_SUPPORTED_GATESET: ops.Gateset = ops.Gateset(ops.Pauli, ops.IdentityGate, ops.Rz, ops.ZPowGate)


class CPhaseGaugeTransformerMM(MultiMomentGaugeTransformer):

    def __init__(self, supported_gates=_SUPPORTED_GATESET):
        super().__init__(target=_TARGET_GATESET, supported_gates=supported_gates)

    def sample_left_moment(self, active_qubits: frozenset[ops.Qid]) -> circuits.Moment:
        return circuits.Moment(
            [
                self.rng.choice(
                    np.array([ops.I, ops.X, ops.Y, ops.Z], dtype=ops.Gate),
                    p=[0.25, 0.25, 0.25, 0.25],
                ).on(q)
                for q in active_qubits
            ]
        )

    def gauge_on_moments(self, moments_to_gauge) -> list[circuits.Moment]:
        active_qubits = circuits.Circuit.from_moments(*moments_to_gauge).all_qubits()
        left_moment = self.sample_left_moment(active_qubits)
        pulled: dict[ops.Qid, _PauliAndZPow] = {
            op.qubits[0]: _PauliAndZPow(pauli=cast(ops.Pauli | ops.IdentityGate, op.gate))
            for op in left_moment
            if op.gate
        }
        ret: list[circuits.Moment] = [left_moment]
        # The loop iterates through each moment of the target block, propagating
        # the `pulled` gauge from left to right. In each iteration, `prev` holds
        # the gauge to the left of the current `moment`, and the loop computes
        # the transformed `moment` and the new `pulled` gauge to its right.
        for moment in moments_to_gauge:
            # Calculate --prev--moment-- ==> --updated_momment--pulled--
            prev = pulled
            pulled = {}
            ops_at_updated_moment: list[ops.Operation] = []
            for op in moment:
                # Pull prev through ops at the moment.
                if op.gate:
                    match op.gate:
                        case ops.CZPowGate():
                            q0, q1 = op.qubits
                            new_gate, pulled[q0], pulled[q1] = _pull_through_single_cphase(
                                op.gate, prev[q0], prev[q1]
                            )
                            ops_at_updated_moment.append(new_gate.on(q0, q1))
                        case ops.Pauli() | ops.IdentityGate():
                            q = op.qubits[0]
                            ops_at_updated_moment.append(op)
                            pulled[q] = prev[q].after_pauli(op.gate)
                        case ops.ZPowGate():
                            q = op.qubits[0]
                            new_zpow, pulled[q] = prev[q].after_zpow(op.gate)
                            ops_at_updated_moment.append(new_zpow.on(q))
                        case _:
                            raise ValueError(f"Gate type {type(op.gate)} is not supported.")
                # Keep the other ops of prev
                for q, gate in prev.items():
                    if q not in pulled:
                        pulled[q] = gate
            ret.append(circuits.Moment(ops_at_updated_moment))
        last_moment = circuits.Moment(
            [gate.to_single_qubit_gate().on(q) for q, gate in pulled.items()]
        )
        ret.append(last_moment)
        return ret
