# Copyright 2024 The Cirq Developers
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

"""A Gauge transformer for SQRT_ISWAP gate."""

from __future__ import annotations

import numpy as np

from cirq import ops
from cirq._doc import document
from cirq.transformers.gauge_compiling.gauge_compiling import (
    ConstantGauge,
    Gauge,
    GaugeSelector,
    GaugeTransformer,
)


class RZRotation(Gauge):
    """Represents a SQRT_ISWAP Gauge composed of Rz rotations.

    The gauge replaces an SQRT_ISWAP gate with either
        0: ───Rz(t)───iSwap───────Rz(-t)───
                        │
        1: ───Rz(t)───iSwap^0.5───Rz(-t)───

    where t is uniformly sampled from [0, 2π).
    """

    def weight(self) -> float:
        return 1.0

    def _rz(self, theta: float) -> ConstantGauge:
        """Returns a SQRT_ISWAP Gauge composed of Rz rotations.

        0: ───Rz(theta)────iSwap───Rz(theta)───
                            │
        1: ───Rz(theta)───iSwap^0.5───Rz(theta)───

        """
        rz = ops.rz(theta)
        n_rz = ops.rz(-theta)
        return ConstantGauge(
            two_qubit_gate=ops.SQRT_ISWAP, pre_q0=rz, pre_q1=rz, post_q0=n_rz, post_q1=n_rz
        )

    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> ConstantGauge:
        return self._rz(prng.random() * 2 * np.pi)


class XYRotation(Gauge):
    """Represents a SQRT_ISWAP Gauge composed of XY rotations.

    The gauge replaces an SQRT_ISWAP gate with either
        0: ───XY(t)───iSwap───────XY(t)───
                        │
        1: ───XY(t)───iSwap^0.5───XY(t)───

    where t is uniformly sampled from [0, 2π) and
        XY(theta) = cos(theta) X + sin(theta) Y
    """

    def weight(self) -> float:
        return 1.0

    def _xy(self, theta: float) -> ops.PhasedXZGate:
        unitary = np.cos(theta) * np.array([[0, 1], [1, 0]]) + np.sin(theta) * np.array(
            [[0, -1j], [1j, 0]]
        )
        return ops.PhasedXZGate.from_matrix(unitary)

    def _xy_gauge(self, theta: float) -> ConstantGauge:
        xy = self._xy(theta)
        return ConstantGauge(
            two_qubit_gate=ops.SQRT_ISWAP, pre_q0=xy, pre_q1=xy, post_q0=xy, post_q1=xy
        )

    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> ConstantGauge:
        return self._xy_gauge(prng.random() * 2 * np.pi)


SqrtISWAPGaugeSelector = GaugeSelector(gauges=[RZRotation(), XYRotation()])

SqrtISWAPGaugeTransformer = GaugeTransformer(
    target=ops.SQRT_ISWAP, gauge_selector=SqrtISWAPGaugeSelector
)
document(
    SqrtISWAPGaugeTransformer,
    r"""A GaugeTransformer that gauges SQRT_ISWAP gates.

    Usage: `SqrtISWAPGaugeTransformer(circuit)` will return a new circuit with all SQRT_ISWAP gates gauged by either the RZRotation gauge or the XYRotation gauge.
    """,
)
