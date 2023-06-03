# Copyright 2023 The Cirq Developers
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
import functools
import numpy as np
from cirq import ops


class GateThatAllocatesAQubit(ops.Gate):
    r"""A gate that applies $Z^\theta$ indirectly through a clean ancilla."""

    def __init__(self, theta: float) -> None:
        super().__init__()
        self._theta = theta

    def _num_qubits_(self):
        return 1

    def _decompose_(self, q):
        anc = ops.NamedQubit("anc")
        yield ops.CX(*q, anc)
        yield (ops.Z**self._theta)(anc)
        yield ops.CX(*q, anc)

    def target_unitary(self) -> np.ndarray:
        return np.array([[1, 0], [0, (-1 + 0j) ** self._theta]])


class GateThatAllocatesTwoQubits(ops.Gate):
    r"""A gate that applies $-j Z \otimes Z$ indirectly through two ancillas."""

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qs):
        q0, q1 = qs
        anc = ops.NamedQubit.range(2, prefix='two_ancillas_')

        yield ops.X(anc[0])
        yield ops.CX(q0, anc[0])
        yield (ops.Y)(anc[0])
        yield ops.CX(q0, anc[0])

        yield ops.CX(q1, anc[1])
        yield (ops.Z)(anc[1])
        yield ops.CX(q1, anc[1])

    @classmethod
    def target_unitary(cls) -> np.ndarray:
        # Unitary = -j Z \otimes Z
        return np.array([[-1j, 0, 0, 0], [0, 1j, 0, 0], [0, 0, 1j, 0], [0, 0, 0, -1j]])


class GateThatDecomposesIntoNGates(ops.Gate):
    r"""Applies $(Z^\theta)^{\otimes_n}$ on work qubits and `subgate` on $n$ borrowable ancillas."""

    def __init__(self, n: int, subgate: ops.Gate, theta: float) -> None:
        super().__init__()
        self._n = n
        self._subgate = subgate
        self._name = str(subgate)
        self._theta = theta

    def _num_qubits_(self) -> int:
        return self._n

    def _decompose_(self, qs):
        ancilla = ops.NamedQubit.range(self._n, prefix=self._name)
        yield self._subgate.on_each(ancilla)
        yield (ops.Z**self._theta).on_each(qs)
        yield self._subgate.on_each(ancilla)

    def target_unitary(self) -> np.ndarray:
        U = np.array([[1, 0], [0, (-1 + 0j) ** self._theta]])
        return functools.reduce(np.kron, [U] * self._n)
