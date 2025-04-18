# Copyright 2019 The Cirq Developers
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

from __future__ import annotations

from typing import Callable, Iterator

import sympy

import cirq
from cirq.interop.quirk.cells.cell import CELL_SIZES, CellMaker, ExplicitOperationsCell


def generate_all_frequency_space_cell_makers() -> Iterator[CellMaker]:
    # Frequency space.
    yield from _family("QFT", lambda n: cirq.QuantumFourierTransformGate(n))
    yield from _family("QFT†", lambda n: cirq.inverse(cirq.QuantumFourierTransformGate(n)))
    yield from _family(
        "PhaseGradient", lambda n: cirq.PhaseGradientGate(num_qubits=n, exponent=0.5)
    )
    yield from _family(
        "PhaseUngradient", lambda n: cirq.PhaseGradientGate(num_qubits=n, exponent=-0.5)
    )
    yield from _family(
        "grad^t",
        lambda n: cirq.PhaseGradientGate(num_qubits=n, exponent=2 ** (n - 1) * sympy.Symbol('t')),
    )
    yield from _family(
        "grad^-t",
        lambda n: cirq.PhaseGradientGate(
            num_qubits=n, exponent=-(2 ** (n - 1)) * sympy.Symbol('t')
        ),
    )


def _family(identifier_prefix: str, gate_maker: Callable[[int], cirq.Gate]) -> Iterator[CellMaker]:
    f = lambda args: ExplicitOperationsCell([gate_maker(len(args.qubits)).on(*args.qubits)])
    yield CellMaker(identifier_prefix, 1, f)
    for i in CELL_SIZES:
        yield CellMaker(identifier_prefix + str(i), i, f)
