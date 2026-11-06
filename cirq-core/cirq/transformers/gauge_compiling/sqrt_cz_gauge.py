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
from typing import TYPE_CHECKING

from cirq import protocols
from cirq.ops import CZ, CZPowGate, Gate, Gateset, S, X
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

    import cirq

_SQRT_CZ = CZPowGate(exponent=0.5)
_ADJ_SQRT_CZ = CZPowGate(exponent=-0.5)
_ADJ_S = S**-1


def _canonical_sqrt_cz(gate: cirq.Gate) -> CZPowGate:
    """Returns CZ**0.5 or CZ**-0.5, whichever `gate` equals up to global phase.

    The transformer's target gateset accepts gates that equal CZ**0.5 or CZ**-0.5 only up to
    global phase (e.g. `cirq.ControlledGate(cirq.S)`, a `CZPowGate` with a global shift, or a
    `MatrixGate`), so the gauge must be chosen with the same notion of equality.

    Raises:
        ValueError: If `gate` equals neither CZ**0.5 nor CZ**-0.5 up to global phase.
    """
    if gate == _SQRT_CZ or protocols.equal_up_to_global_phase(gate, _SQRT_CZ):
        return _SQRT_CZ
    if gate == _ADJ_SQRT_CZ or protocols.equal_up_to_global_phase(gate, _ADJ_SQRT_CZ):
        return _ADJ_SQRT_CZ
    raise ValueError(
        f"SqrtCZGauge supports only gates equal to CZ**0.5 or CZ**-0.5 up to global phase, "
        f"got {gate!r}."
    )


class SqrtCZGauge(Gauge):

    def weight(self) -> float:
        return 3.0

    def sample(self, gate: cirq.Gate, prng: np.random.Generator) -> ConstantGauge:
        canonical = _canonical_sqrt_cz(gate)
        if prng.choice([True, False]):
            return ConstantGauge(two_qubit_gate=gate)
        correction = S if canonical == _SQRT_CZ else _ADJ_S
        # Keep the gate's own representation when it can be inverted; a gate that only
        # implements `_unitary_` cannot, so fall back to the canonical inverse.
        inverse = protocols.pow(gate, -1, None)
        if inverse is None:
            inverse = canonical**-1
        swap_qubits = prng.choice([True, False])
        if swap_qubits:
            return ConstantGauge(
                pre_q1=X, post_q1=X, post_q0=correction, two_qubit_gate=inverse, swap_qubits=True
            )
        else:
            return ConstantGauge(pre_q0=X, post_q0=X, post_q1=correction, two_qubit_gate=inverse)


def _symbolize_as_cz_pow(
    two_qubit_gate: Gate, symbols: Sequence[sympy.Symbol]
) -> tuple[Gate, dict[str, Real]]:
    """Symbolizes a CZPowGate, or a gate equal to CZ**±0.5 up to global phase, as CZ**symbol."""

    if isinstance(two_qubit_gate, CZPowGate) and isinstance(two_qubit_gate.exponent, Real):
        exponent = two_qubit_gate.exponent
    else:
        try:
            exponent = _canonical_sqrt_cz(two_qubit_gate).exponent
        except ValueError:
            raise ValueError("Can't symbolize non-CZPowGate as CZ**symbol.") from None
    return (CZ ** symbols[0], {str(symbols[0]): exponent})


SqrtCZGaugeTransformer = GaugeTransformer(
    target=Gateset(_SQRT_CZ, _SQRT_CZ**-1),
    gauge_selector=GaugeSelector(gauges=[SqrtCZGauge()]),
    two_qubit_gate_symbolizer=TwoQubitGateSymbolizer(
        symbolizer_fn=_symbolize_as_cz_pow, n_symbols=1
    ),
)
