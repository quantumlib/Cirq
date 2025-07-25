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

from __future__ import annotations

import sympy
import tunits as tu

import cirq
from cirq_google.study import symbol_util as su


class WaitGateWithUnit(cirq.WaitGate):
    """A wrapper on top of WaitGate that can work with units."""

    def __init__(
        self,
        duration: su.ValueOrSymbol,
        num_qubits: int | None = None,
        qid_shape: tuple[int, ...] | None = None,
    ):
        if not isinstance(duration, su.ValueOrSymbol):
            raise ValueError("The duration must either be a tu.Value or a sympy.Symbol.")
        # Override the original duration
        self._duration: su.ValueOrSymbol = duration  # type: ignore[assignment]

        # The rest is copy-pasted from WaitGate. We just cannot use
        # super().__init__ because of the duration.
        if qid_shape is None:
            if num_qubits is None:
                # Assume one qubit for backwards compatibility
                qid_shape = (2,)
            else:
                qid_shape = (2,) * num_qubits
        if num_qubits is None:
            num_qubits = len(qid_shape)
        if not qid_shape:
            raise ValueError('Waiting on an empty set of qubits.')
        if num_qubits != len(qid_shape):
            raise ValueError('len(qid_shape) != num_qubits')
        self._qid_shape = qid_shape

    @property
    def duration(self) -> sympy.Symbol | cirq.Duration:
        if isinstance(self._duration, sympy.Symbol):
            return self._duration
        return cirq.Duration(nanos=self._duration[tu.ns])

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> WaitGateWithUnit:
        if isinstance(self._duration, sympy.Symbol):
            _duration = su.direct_symbol_replacement(self._duration, resolver)
            return WaitGateWithUnit(_duration, qid_shape=self._qid_shape)
        return self
