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

"""A Gauge transformer for ISWAP gate."""

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
    """Represents an ISWAP Gauge composed of Rz rotations.

    The gauge replaces an ISWAP gate with either
        0: ───Rz(t)──────iSwap───Rz(sgn*t)───
                            │
        1: ───Rz(-sgn*t)───iSwap───Rz(-t)───

    where t is uniformly sampled from [0, 2π) and sgn is uniformly sampled from {-1, 1}.
    """

    def weight(self) -> float:
        return 2.0

    def _rz(self, theta, sgn: int) -> ConstantGauge:
        """Returns an ISWAP Gauge composed of Rz rotations.

        0: ───Rz(theta)──────iSwap───Rz(sgn*theta)───
                                │
        1: ───Rz(-sgn*theta)───iSwap───Rz(-theta)───

        """
        flip_diangonal = sgn == -1
        rz = ops.rz(theta)
        n_rz = ops.rz(-theta)
        return ConstantGauge(
            two_qubit_gate=ops.ISWAP,
            pre_q0=rz,
            pre_q1=n_rz if flip_diangonal else rz,
            post_q0=rz if flip_diangonal else n_rz,
            post_q1=n_rz,
        )

    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> ConstantGauge:
        theta = prng.random() * 2 * np.pi
        return self._rz(theta, prng.choice([-1, 1]))


class XYRotation(Gauge):
    """Represents an ISWAP Gauge composed of XY rotations.

    The gauge replaces an ISWAP gate with either
        0: ───XY(a)───iSwap───XY(b)───
                        │
        1: ───XY(b)───iSwap───XY(a)───

    where a and b are uniformly sampled from [0, 2π) and XY is a single-qubit rotation defined as
        XY(theta) = cos(theta) X + sin(theta) Y
    """

    def weight(self) -> float:
        return 2.0

    def _xy(self, theta: float) -> ops.PhasedXZGate:
        unitary = np.cos(theta) * np.array([[0, 1], [1, 0]]) + np.sin(theta) * np.array(
            [[0, -1j], [1j, 0]]
        )
        return ops.PhasedXZGate.from_matrix(unitary)

    def _xy_gauge(self, a: float, b: float) -> ConstantGauge:
        xy_a = self._xy(a)
        xy_b = self._xy(b)
        return ConstantGauge(
            two_qubit_gate=ops.ISWAP, pre_q0=xy_a, pre_q1=xy_b, post_q0=xy_b, post_q1=xy_a
        )

    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> ConstantGauge:
        a = prng.random() * 2 * np.pi
        if prng.choice([0, 1]):
            return self._xy_gauge(a, a)
        else:
            b = prng.random() * 2 * np.pi
            return self._xy_gauge(a, b)


ISWAPGaugeTransformer = GaugeTransformer(
    target=ops.ISWAP, gauge_selector=GaugeSelector(gauges=[RZRotation(), XYRotation()])
)
document(
    ISWAPGaugeTransformer,
    r"""A GaugeTransformer that gauges ISWAP gates.

    Usage: `ISWAPGaugeTransformer(circuit)` will return a new circuit with all ISWAP gates gauged by either the RZRotation gauge or the XYRotation gauge.
    """,
)
