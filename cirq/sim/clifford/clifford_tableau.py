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

from typing import Any, Dict, List
import numpy as np

import cirq
from cirq import protocols
from cirq.ops.dense_pauli_string import DensePauliString
from cirq.value import big_endian_int_to_digits


class CliffordTableau():
    """ Tableau representation of a stabilizer state
    (based on Aaronson and Gottesman 2006).

    The tableau stores the stabilizer generators of
    the state using three binary arrays: xs, zs, and rs.

    Each row of the arrays represents a Pauli string, P, that is
    an eigenoperator of the state vector with eigenvalue one: P|psi> = |psi>.
    """

    def __init__(self, num_qubits, initial_state: int = 0):
        """Initializes CliffordTableau
        Args:
            num_qubits: The number of qubits in the system.
            initial_state: The computational basis representation of the
                state as a big endian int.
            """
        self.n = num_qubits

        self.rs = np.zeros(2 * self.n + 1, dtype=bool)

        for (i, val) in enumerate(
                big_endian_int_to_digits(initial_state,
                                         digit_count=num_qubits,
                                         base=2)):
            self.rs[self.n + i] = bool(val)

        self.xs = np.zeros((2 * self.n + 1, self.n), dtype=bool)
        self.zs = np.zeros((2 * self.n + 1, self.n), dtype=bool)

        for i in range(self.n):
            self.xs[i, i] = True
            self.zs[self.n + i, i] = True

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['n', 'rs', 'xs', 'zs'])

    @classmethod
    def _from_json_dict_(cls, n, rs, xs, zs, **kwargs):
        state = cls(n)
        state.rs = rs
        state.xs = xs
        state.zs = zs
        return state

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            # coverage: ignore
            return NotImplemented
        return (self.n == other.n and np.array_equal(self.rs, other.rs) and
                np.array_equal(self.xs, other.xs) and
                np.array_equal(self.zs, other.zs))

    def copy(self) -> 'CliffordTableau':
        state = CliffordTableau(self.n)
        state.rs = self.rs.copy()
        state.xs = self.xs.copy()
        state.zs = self.zs.copy()
        return state

    def __repr__(self) -> str:
        stabilizers = ", ".join([repr(stab) for stab in self.stabilizers()])
        return f'stabilizers: [{stabilizers}]'

    def __str__(self) -> str:
        string = ''

        for i in range(self.n, 2 * self.n):
            string += '- ' if self.rs[i] else '+ '

            for k in range(0, self.n):
                if self.xs[i, k] & (not self.zs[i, k]):
                    string += 'X '
                elif (not self.xs[i, k]) & self.zs[i, k]:
                    string += 'Z '
                elif self.xs[i, k] & self.zs[i, k]:
                    string += 'Y '
                else:
                    string += 'I '

            if i < 2 * self.n - 1:
                string += '\n'

        return string

    def _str_full_(self) -> str:
        string = ''

        string += 'stable' + ' ' * max(self.n * 2 - 3, 1)
        string += '| destable\n'
        string += '-' * max(7, self.n * 2 + 3) + '+' + '-' * max(
            10, self.n * 2 + 4) + '\n'

        for j in range(self.n):
            for i in [j + self.n, j]:
                string += '- ' if self.rs[i] else '+ '

                for k in range(0, self.n):
                    if self.xs[i, k] & (not self.zs[i, k]):
                        string += 'X%d' % k
                    elif (not self.xs[i, k]) & self.zs[i, k]:
                        string += 'Z%d' % k
                    elif self.xs[i, k] & self.zs[i, k]:
                        string += 'Y%d' % k
                    else:
                        string += '  '

                if i == j + self.n:
                    string += ' ' * max(0, 4 - self.n * 2) + ' | '

            string += '\n'

        return string

    def _rowsum(self, q1, q2):
        """Implements the "rowsum" routine defined by
        Aaronson and Gottesman.
        Multiplies the stabilizer in row q1 by the stabilizer in row q2."""

        def g(x1, z1, x2, z2):
            if not x1 and not z1:
                return 0
            elif x1 and z1:
                return int(z2) - int(x2)
            elif x1 and not z1:
                return int(z2) * (2 * int(x2) - 1)
            else:
                return int(x2) * (1 - 2 * int(z2))

        r = 2 * int(self.rs[q1]) + 2 * int(self.rs[q2])
        for j in range(self.n):
            r += g(self.xs[q2, j], self.zs[q2, j], self.xs[q1, j],
                   self.zs[q1, j])

        r %= 4

        self.rs[q1] = bool(r)

        self.xs[q1, :] ^= self.xs[q2, :]
        self.zs[q1, :] ^= self.zs[q2, :]

    def _row_to_dense_pauli(self, i: int) -> DensePauliString:
        """
        Args:
            i: index of the row in the tableau.
        Returns:
            A DensePauliString representing the row. The length of the string
            is equal to the total number of qubits and each character
            represents the effective single Pauli operator on that qubit. The
            overall phase is captured in the coefficient.
        """
        coefficient = -1 if self.rs[i] else 1
        pauli_mask = ""

        for k in range(self.n):
            if self.xs[i, k] & (not self.zs[i, k]):
                pauli_mask += "X"
            elif (not self.xs[i, k]) & self.zs[i, k]:
                pauli_mask += "Z"
            elif self.xs[i, k] & self.zs[i, k]:
                pauli_mask += "Y"
            else:
                pauli_mask += "I"
        return cirq.DensePauliString(pauli_mask, coefficient=coefficient)

    def stabilizers(self) -> List[DensePauliString]:
        """Returns the stabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that S_i |psi> = |psi> """
        return [self._row_to_dense_pauli(i) for i in range(self.n, 2 * self.n)]

    def destabilizers(self) -> List[DensePauliString]:
        """Returns the destabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that along with the stabilizer
        generators above generate the full Pauli group on n qubits."""
        return [self._row_to_dense_pauli(i) for i in range(0, self.n)]

    def _measure(self, q, prng: np.random.RandomState):
        """ Performs a projective measurement on the q'th qubit.

        Returns: the result (0 or 1) of the measurement.
        """
        is_commuting = True
        for i in range(self.n, 2 * self.n):
            if self.xs[i, q]:
                p = i
                is_commuting = False
                break

        if is_commuting:
            self.xs[2 * self.n, :] = False
            self.zs[2 * self.n, :] = False
            self.rs[2 * self.n] = False

            for i in range(self.n):
                if self.xs[i, q]:
                    self._rowsum(2 * self.n, self.n + i)
            return int(self.rs[2 * self.n])

        else:
            for i in range(2 * self.n):
                if i != p and self.xs[i, q]:
                    self._rowsum(i, p)

            self.xs[p - self.n, :] = self.xs[p, :]
            self.zs[p - self.n, :] = self.zs[p, :]
            self.rs[p - self.n] = self.rs[p]

            self.xs[p, :] = False
            self.zs[p, :] = False

            self.zs[p, q] = True

            self.rs[p] = bool(prng.randint(2))

            return int(self.rs[p])
