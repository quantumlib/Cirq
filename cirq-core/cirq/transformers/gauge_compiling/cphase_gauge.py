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

import cirq.transformers.gauge_compiling.sqrt_cz_gauge as sqrt_cz_gauge

from cirq.transformers.gauge_compiling.gauge_compiling import (
    GaugeTransformer,
    GaugeSelector,
    ConstantGauge,
    Gauge,
    TwoQubitGateSymbolizer,
)
from cirq import ops
import numpy as np


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
