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

"""A Gauge Transformer for the cphase gate."""

from __future__ import annotations

from typing import List

import numpy as np

import cirq.transformers.gauge_compiling.sqrt_cz_gauge as sqrt_cz_gauge
from cirq import circuits, ops
from cirq.transformers.gauge_compiling.gauge_compiling import (
    ConstantGauge,
    Gauge,
    GaugeSelector,
    GaugeTransformer,
    TwoQubitGateSymbolizer,
)


class CPhasePauliGauge(Gauge):
    """Gauges for the cphase gate (CZPowGate).

    We identify 16 distinct gauges, corresponding to the 16 two-qubit Pauli operators that can be
    inserted before the cphase gate. When an anticommuting gate is inserted, the cphase angle is
    negated (or equivalently, the exponent of the CZPowGate is negated), so both postive and
    negative angles should be calibrated to use this.
    """

    def weight(self) -> float:
        return 1.0

    def _get_new_post(self, exponent: float, pre: ops.Gate) -> ops.Gate:
        """Identify the new single-qubit gate that needs to be inserted in the case that both pre
        gates are X or Y.

        Args:
            exponent: The exponent of the CZPowGate that is getting transformed.
            pre: The gate (X or Y) that is inserted on a given qubit.

        Returns:
            The single-qubit gate to insert after the CZPowGate on the same qubit.

        Raises:
            ValueError: If pre is not X or Y.
        """
        if pre == ops.X:
            return ops.PhasedXPowGate(exponent=1, phase_exponent=exponent / 2)
        elif pre == ops.Y:
            return ops.PhasedXZGate.from_zyz_exponents(z0=-exponent, y=1, z1=0)
        else:
            raise ValueError("pre should be cirq.X or cirq.Y")  # pragma: no cover

    def _get_constant_gauge(
        self, gate: ops.CZPowGate, pre_q0: ops.Gate, pre_q1: ops.Gate
    ) -> ConstantGauge:
        """Get the ConstantGauge corresponding to a given pre_q0 and pre_q1.

        Args:
            gate: The particular cphase gate to transform.
            pre_q0: The Pauli (I, X, Y, or Z) to insert before the cphase gate on q0.
            pre_q1: The Pauli (I, X, Y, or Z) to insert before the cphase gate on q1.

        Returns:
            The ConstantGauge implementing the given transformation.

        Raises:
            ValueError: If pre_q0 and pre_q1 are not both in {I, X, Y, Z}.
        """

        exponent = gate.exponent
        commuting_paulis = {ops.I, ops.Z}
        anticommuting_paulis = {ops.X, ops.Y}
        if pre_q0 in commuting_paulis and pre_q1 in commuting_paulis:
            return ConstantGauge(
                two_qubit_gate=gate, pre_q0=pre_q0, pre_q1=pre_q1, post_q0=pre_q0, post_q1=pre_q1
            )
        elif pre_q0 in anticommuting_paulis and pre_q1 in commuting_paulis:
            new_gate = ops.CZ ** (-exponent)
            post_q1 = ops.Z ** (exponent) if pre_q1 == ops.I else ops.Z ** (1 + exponent)
            return ConstantGauge(
                two_qubit_gate=new_gate,
                pre_q0=pre_q0,
                pre_q1=pre_q1,
                post_q0=pre_q0,
                post_q1=post_q1,
            )
        elif pre_q0 in commuting_paulis and pre_q1 in anticommuting_paulis:
            new_gate = ops.CZ ** (-exponent)
            post_q0 = ops.Z ** (exponent) if pre_q0 == ops.I else ops.Z ** (1 + exponent)
            return ConstantGauge(
                two_qubit_gate=new_gate,
                pre_q0=pre_q0,
                pre_q1=pre_q1,
                post_q0=post_q0,
                post_q1=pre_q1,
            )
        elif pre_q0 in anticommuting_paulis and pre_q1 in anticommuting_paulis:
            return ConstantGauge(
                two_qubit_gate=gate,
                pre_q0=pre_q0,
                pre_q1=pre_q1,
                post_q0=self._get_new_post(exponent, pre_q0),
                post_q1=self._get_new_post(exponent, pre_q1),
            )
        else:
            raise ValueError("pre_q0 and pre_q1 should be X, Y, Z, or I")  # pragma: no cover

    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> ConstantGauge:
        """Sample the 16 cphase gauges at random.

        Args:
            gate: The CZPowGate to transform.
            prng: The pseudorandom number generator.

        Returns:
            A ConstantGauge implementing the transformation.

        Raises:
            TypeError: if gate is not a CZPowGate
        """

        if not type(gate) == ops.CZPowGate:
            raise TypeError("gate must be a CZPowGate")  # pragma: no cover
        pre_q0, pre_q1 = prng.choice(np.array([ops.I, ops.X, ops.Y, ops.Z]), size=2, replace=True)
        return self._get_constant_gauge(gate, pre_q0, pre_q1)


CPhaseGaugeSelector = GaugeSelector(gauges=[CPhasePauliGauge()])

CPhaseGaugeTransformer = GaugeTransformer(
    target=ops.Gateset(ops.CZPowGate),
    gauge_selector=CPhaseGaugeSelector,
    two_qubit_gate_symbolizer=TwoQubitGateSymbolizer(
        symbolizer_fn=sqrt_cz_gauge._symbolize_as_cz_pow, n_symbols=1
    ),
)


class _PhasedXYAndRz:
    """In pulling through, one qubit gate can be represented by a Pauli and an Rz gate.

    The order is --(X|Y|I)--Rz(rad)--phase--.
    """

    pauli: ops.X | ops.Y | ops.I
    rz_rads: float
    phase_exp: float  # phase of the qubit is e^{i*phase_exp*pi}

    def __init__(
        self, pauli: ops.Pauli | ops.I = ops.I, rz_rads: float = 0, phase_exp: float = 0
    ) -> None:
        if pauli == ops.Z:  # Merge Z gates to Rz where Z = Rz(π) * e^{iπ/2}
            self.pauli = ops.I
            self.rz_rads = rz_rads + np.pi
            self.phase_exp = phase_exp + 0.5
        else:
            self.pauli = pauli
            self.rz_rads = rz_rads
            self.phase_exp = phase_exp

    def _merge_left_rz(self, rads: float):
        """Merges Rz(rad) from left."""
        if self.pauli == ops.I:
            self.rz_rads += rads
        else:
            self.rz_rads -= rads

    def _merge_right_rz(self, rads: float):
        """Merges Rz(rads) from right."""
        self.rz_rads += rads

    def _merge_left_xy(self, other: ops.X | ops.Y):
        """Merges --(X|Y)--self--."""
        if self.pauli == other:
            self.pauli = ops.I
            return
        if self.pauli == ops.I:
            self.pauli = other
            return
        if (other, self.pauli) == (ops.X, ops.Y):
            # -X--Y-  ==>  --Rz(pi)--
            self.pauli = ops.I
            self.rz_rads += np.pi
            return
        if (other, self.pauli) == (ops.Y, ops.X):
            # -Y--X-  ==>  --Rz(-pi)--
            self.pauli = ops.I
            self.rz_rads -= np.pi
            return

    def _merge_right_xy(self, other: ops.X | ops.Y):
        """Merges --self--(X|Y)--."""
        self.rz_rads *= -1
        if self.pauli == other:
            self.pauli = ops.I
            return
        if self.pauli == ops.I:
            self.pauli = other
            return
        if (self.pauli, other) == (ops.X, ops.Y):
            # -X--Y-  ==>  --Rz(pi)--
            self.pauli = ops.I
            self.rz_rads += np.pi
            return
        if (self.pauli, other) == (ops.Y, ops.X):
            # -X--Y-  ==>  --Rz(-pi)--
            self.pauli = ops.I
            self.rz_rads -= np.pi
            return

    def merge_left(self, other: _PhasedXYAndRz) -> None:
        """Inplace merge other from left."""
        self._merge_left_rz(other.rz_rads)
        self.phase_exp += other.phase_exp
        if other.pauli != ops.I:
            self._merge_left_xy(other.pauli)

    def merge_right(self, other: _PhasedXYAndRz) -> None:
        """Inplace merge other from right."""
        self.phase_exp += other.phase_exp
        if other.pauli != ops.I:
            self._merge_right_xy(other.pauli)
        self._merge_right_rz(other.rz_rads)

    def after_cphase(
        self, cphase: ops.CZPowGate
    ) -> tuple[ops.CZPowGate, _PhasedXYAndRz, _PhasedXYAndRz]:
        """Pull self through cphase.

        Returns:
            updated cphase gate, pull_through of this qubit, pull_through of the other qubit.
        """
        match self.pauli:
            case ops.I:
                return cphase, self, _PhasedXYAndRz()
            case _:  # ops.X | ops.Y:
                # Taking input0 with X gate as an example:
                # 0: ─X─Rz(t)─phase─@──────      0: ─X──@─────Rz(t)──phase─
                #                   │       ==>         │
                # 1: ───────────────@^exp──      1: ────@^exp──────────────
                #      0: ─@──────X────Rz(t)───phase─────────
                # ==>      │
                #      1: ─@^-exp─Rz(exp pi)─e^{-exp pi/2 i}─
                # where rad = -exp * pi.
                # Similarly for X|Y on qubit 0/1, the result is always flipping cphase and
                # add an extra Rz rotation on the other qubit.
                return (
                    cphase**-1,
                    self,
                    _PhasedXYAndRz(rz_rads=cphase.exponent * np.pi, phase_exp=-cphase.exponent / 2),
                )

    def __str__(self) -> str:
        return f"─{self.pauli}──Rz({self.rz_rads})──phase(e^{{i{self.phase_exp}π}})─"

    def __eq__(self, other: _PhasedXYAndRz) -> bool:
        return (
            self.pauli == other.pauli
            and np.isclose(self.rz_rads, other.rz_rads, atol=1e-10)
            and np.isclose(self.phase_exp, other.phase_exp, atol=1e-10)
        )

    def to_single_gate(self) -> ops.PhasedXZGate | ops.ZPowGate:
        if self.pauli == ops.I:
            rz_rads = self.rz_rads
            if np.isclose(self.rz_rads, 0, atol=1e-2):
                rz_rads = self.rz_rads + 4 * np.pi
            return ops.ZPowGate(
                exponent=rz_rads / np.pi, global_shift=np.pi * self.phase_exp / rz_rads - 0.5
            )
        if self.pauli == ops.X:
            return ops.PhasedXZGate(
                x_exponent=1,
                z_exponent=2 * self.phase_exp,
                axis_phase_exponent=self.rz_rads / 2 / np.pi - self.phase_exp,
            )
        if self.pauli == ops.Y:
            return ops.PhasedXZGate(
                x_exponent=1,
                z_exponent=2 * self.phase_exp,
                axis_phase_exponent=1 / 2 - self.phase_exp + self.rz_rads / 2 / np.pi,
            )


def _pull_through_single_cphase(
    cphase: ops.CZPowGate, input0: _PhasedXYAndRz, input1: _PhasedXYAndRz
) -> tuple[ops.CZPowGate, _PhasedXYAndRz, _PhasedXYAndRz]:
    """Pulls input0 and input1 through a CZPowGate.
    Input:
    0: ─(input0=P0──Rz0──phase0)─@─────
                                 │
    1: ─(input1=P1──Rz1──phase1)─@^exp─
    Output:
    0: ─@────────(output0=P0'──Rz0'──phase0')─
        │
    1: ─@^+/-exp─(output1=P1'──Rz1'──phase1')─
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


def _multi_moment_pull_through(
    moments: List[circuits.Moment], rng: np.random.Generator
) -> List[circuits.Moment]:
    """TO FILL."""
    all_qubits = [q for q in circuits.Circuit(moments).all_qubits()]
    if not all_qubits:
        return moments
    if not any(isinstance(op.gate, ops.CZPowGate) for moment in moments for op in moment):
        return moments

    left_moment = circuits.Moment(
        [rng.choice([ops.I, ops.X, ops.Y, ops.Z]).on(q) for q in all_qubits]
    )
    prev: map[ops.Qid, ops.Gate] = {
        op.qubits[0]: _PhasedXYAndRz(pauli=op.gate) for op in left_moment
    }

    new_moments: List[circuits.Moment] = [left_moment]

    pulled: map[ops.Qid, ops.Gate]
    for moment in moments:
        pulled = {}
        new_moment: List[ops.Operation] = []
        for op in moment:
            if op.gate and (isinstance(op.gate, ops.CZPowGate)):
                q0, q1 = op.qubits
                cphase_gate, pulled[q0], pulled[q1] = _pull_through_single_cphase(
                    op.gate, prev[q0], prev[q1]
                )
                new_moment.append(cphase_gate.on(q0, q1))
            elif op.gate and isinstance(op.gate, ops.ZPowGate):
                q = op.qubits[0]
                pulled[q] = prev[q]
                pulled[q].merge_right(_PhasedXYAndRz(rz_rads=op.gate.exponent * np.pi))
                # Don't need to add the op in the new_moment as it is already merged into pulled.
            else:
                new_moment.append(op)
        for q in all_qubits:
            if q not in pulled:
                pulled[q] = prev[q]
        prev = pulled
        new_moments.append(new_moment)

    last_moment = circuits.Moment([pulled[q].to_single_gate().on(q) for q in all_qubits])

    new_moments.append(last_moment)

    return new_moments


# Multi-moments pull through version of CZGaugeTransformer
CPhaseGaugeTransformerMM = GaugeTransformer(
    target=ops.Gateset(ops.CZPowGate, ops.ZPowGate),
    gauge_selector=CPhaseGaugeSelector,
    multi_moment_pull_thourgh_fn=_multi_moment_pull_through,
)
