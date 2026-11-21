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

"""A Gauge transformer for CZ**0.5 and CZ**-0.5 gates."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import cast, TYPE_CHECKING

from cirq import ops
from cirq.transformers.gauge_compiling.gauge_compiling import (
    ConstantGauge,
    Gauge,
    GaugeSelector,
    GaugeTransformer,
    TwoQubitGateSymbolizer,
)

if TYPE_CHECKING:
    import numpy as np
    import sympy

_SQRT_CZ = ops.CZPowGate(exponent=0.5)
_ADJ_SQRT_CZ = ops.CZPowGate(exponent=-0.5)
_GATESET_SQRT_CZ = ops.Gateset(_SQRT_CZ)
_GATESET_ADJ_SQRT_CZ = ops.Gateset(_ADJ_SQRT_CZ)
_ADJ_S = ops.S**-1


class SqrtCZGauge(Gauge):

    def weight(self) -> float:
        return 3.0

    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> ConstantGauge:
        if prng.choice([True, False]):
            return ConstantGauge(two_qubit_gate=gate)
        swap_qubits = prng.choice([True, False])
        if swap_qubits:
            return ConstantGauge(
                pre_q1=ops.X,
                post_q1=ops.X,
                post_q0=ops.S if gate in _GATESET_SQRT_CZ else _ADJ_S,
                two_qubit_gate=gate**-1,
                swap_qubits=True,
            )
        else:
            return ConstantGauge(
                pre_q0=ops.X,
                post_q0=ops.X,
                post_q1=ops.S if gate in _GATESET_SQRT_CZ else _ADJ_S,
                two_qubit_gate=gate**-1,
            )


def _symbolize_as_cz_pow(
    two_qubit_gate: ops.Gate, symbols: Sequence[sympy.Symbol]
) -> tuple[ops.Gate, dict[str, Real]]:
    """Symbolizes a CZPowGate, or a gate equal to CZ**±0.5 up to global phase, as CZ**symbol."""
    if isinstance(two_qubit_gate, ops.CZPowGate) and isinstance(two_qubit_gate.exponent, Real):
        exponent = two_qubit_gate.exponent
    elif two_qubit_gate in _GATESET_SQRT_CZ:
        exponent = cast(Real, 0.5)
    elif two_qubit_gate in _GATESET_ADJ_SQRT_CZ:
        exponent = cast(Real, -0.5)
    else:
        raise ValueError("Can't symbolize non-CZPowGate as CZ**symbol.")
    return (ops.CZ ** symbols[0], {str(symbols[0]): exponent})


SqrtCZGaugeTransformer = GaugeTransformer(
    target=ops.Gateset(_SQRT_CZ, _ADJ_SQRT_CZ),
    gauge_selector=GaugeSelector(gauges=[SqrtCZGauge()]),
    two_qubit_gate_symbolizer=TwoQubitGateSymbolizer(
        symbolizer_fn=_symbolize_as_cz_pow, n_symbols=1
    ),
)
