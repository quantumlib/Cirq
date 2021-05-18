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
from cirq.value import big_endian_int_to_digits


class CliffordTableau:
    """Tableau representation of a stabilizer state
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

        # The last row (`2n+1`-th row) is the scratch row used in _measurement
        # computation process only. It should not be exposed to external usage.
        self._rs = np.zeros(2 * self.n + 1, dtype=bool)

        for (i, val) in enumerate(
            big_endian_int_to_digits(initial_state, digit_count=num_qubits, base=2)
        ):
            self._rs[self.n + i] = bool(val)

        self._xs = np.zeros((2 * self.n + 1, self.n), dtype=bool)
        self._zs = np.zeros((2 * self.n + 1, self.n), dtype=bool)

        for i in range(self.n):
            self._xs[i, i] = True
            self._zs[self.n + i, i] = True

    @property
    def xs(self) -> np.array:
        return self._xs[:-1, :]

    @xs.setter
    def xs(self, new_xs: np.array) -> None:
        assert np.shape(new_xs) == (2 * self.n, self.n)
        self._xs[:-1, :] = np.array(new_xs).astype(bool)

    @property
    def zs(self) -> np.array:
        return self._zs[:-1, :]

    @zs.setter
    def zs(self, new_zs: np.array) -> None:
        assert np.shape(new_zs) == (2 * self.n, self.n)
        self._zs[:-1, :] = np.array(new_zs).astype(bool)

    @property
    def rs(self) -> np.array:
        return self._rs[:-1]

    @rs.setter
    def rs(self, new_rs: np.array) -> None:
        assert np.shape(new_rs) == (2 * self.n,)
        self._rs[:-1] = np.array(new_rs).astype(bool)

    def matrix(self) -> np.array:
        """Returns the 2n * 2n matrix representation of the Clifford tableau."""
        return np.concatenate([self.xs, self.zs], axis=1)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['n', 'rs', 'xs', 'zs'])

    @classmethod
    def _from_json_dict_(cls, n, rs, xs, zs, **kwargs):
        state = cls(n)
        state.rs = np.array(rs).astype(bool)
        state.xs = np.array(xs).astype(bool)
        state.zs = np.array(zs).astype(bool)
        return state

    def _validate(self) -> bool:
        """Check if the Clifford Tabluea satisfies the symplectic property."""
        table = np.concatenate([self.xs, self.zs], axis=1)
        perm = list(range(self.n, 2 * self.n)) + list(range(self.n))
        skew_eye = np.eye(2 * self.n, dtype=int)[perm]
        return np.array_equal(np.mod(table.T.dot(skew_eye).dot(table), 2), skew_eye)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            # coverage: ignore
            return NotImplemented
        return (
            self.n == other.n
            and np.array_equal(self.rs, other.rs)
            and np.array_equal(self.xs, other.xs)
            and np.array_equal(self.zs, other.zs)
        )

    def __copy__(self) -> 'CliffordTableau':
        return self.copy()

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
        string += '-' * max(7, self.n * 2 + 3) + '+' + '-' * max(10, self.n * 2 + 4) + '\n'

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

    def then(self, second: 'CliffordTableau') -> 'CliffordTableau':
        """Returns a composed CliffordTableau of this tableau and the second tableau.

        Then composed tableau is equal to (up to global phase) the composed
        unitary operation of the two tableaux, i.e. equivalent to applying the unitary
        operation of this CliffordTableau then applying the second one.

        Args:
            second: The second CliffordTableau to compose with.

        Returns:
            The composed CliffordTableau.

        Raises:
            TypeError: If the type of second is not CliffordTableau.
            ValueError: If the number of qubits in the second tableau mismatch with
                this tableau.
        """
        if not isinstance(second, CliffordTableau):
            raise TypeError("The type for second tableau must be the CliffordTableau type")
        if self.n != second.n:
            raise ValueError(
                f"Mismatched number of qubits of two tableaux: {self.n} vs {second.n}."
            )

        # Convert the underlying data type from bool to int for easier numerical computation.
        m1 = self.matrix().astype(int)
        m2 = second.matrix().astype(int)

        p1 = self.rs.astype(int)
        p2 = second.rs.astype(int)

        merged_m = np.mod(m1.dot(m2), 2)
        phase = np.mod(p1 + m1.dot(p2), 2)

        num_ys1 = np.sum(m1[:, : self.n] * m1[:, self.n :], axis=1)
        num_ys2 = np.sum(m2[:, : self.n] * m2[:, self.n :], axis=1)
        # We need more phase correction for expanding Y to XZ and swapping Z_iX_i order.
        # One key property is X_i is only anti-commute with Z_j i.f.f. i = j and X_i
        # commute with all X_j. Hence, we can calculate the phase on X/Z_i independently from
        # X/Z_j, which is corresponding to element wise computation in differnt columns.
        # This is the basis for the vectorization.
        for k in range(2 * self.n):
            swap_phase = 0  # The value mod 4 represents number of i factors.
            prev_row_sum = np.zeros([2 * self.n])
            swap_phase += num_ys1[k]  # Y gate => iXZ
            for i, v in enumerate(m1[k]):
                if v == 0:
                    continue
                swap_phase += num_ys2[i]  # Y gate => iXZ
                # swapping Z_i with X_i adds a -1 phase, which is 2j
                swap_phase += 2 * np.sum(m2[i, : self.n] * prev_row_sum[self.n :])
                prev_row_sum += m2[i]
            prev_row_sum = np.mod(prev_row_sum, 2)
            swap_phase -= np.sum(prev_row_sum[: self.n] * prev_row_sum[self.n :])  # XZ => -iY

            # Adding the correction phase. Note original phase is True/False for [+1, -1] phase.
            # while the swap_phase mod 4 is for [1, i, -1, -i]. At this stage,
            # the swap_phase must be either +1 or -1 guaranteed by the symplectic property.
            phase[k] = (phase[k] + (swap_phase % 4) / 2) % 2

        merged_tableau = CliffordTableau(num_qubits=self.n)
        merged_tableau.xs = merged_m[:, : self.n]
        merged_tableau.zs = merged_m[:, self.n :]
        merged_tableau.rs = phase
        return merged_tableau

    def __matmul__(self, second: 'CliffordTableau'):
        if not isinstance(second, CliffordTableau):
            return NotImplemented
        return second.then(self)

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

        r = 2 * int(self._rs[q1]) + 2 * int(self._rs[q2])
        for j in range(self.n):
            r += g(self._xs[q2, j], self._zs[q2, j], self._xs[q1, j], self._zs[q1, j])

        r %= 4

        self._rs[q1] = bool(r)

        self._xs[q1, :] ^= self._xs[q2, :]
        self._zs[q1, :] ^= self._zs[q2, :]

    def _row_to_dense_pauli(self, i: int) -> 'cirq.DensePauliString':
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

    def stabilizers(self) -> List['cirq.DensePauliString']:
        """Returns the stabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that S_i |psi> = |psi>"""
        return [self._row_to_dense_pauli(i) for i in range(self.n, 2 * self.n)]

    def destabilizers(self) -> List['cirq.DensePauliString']:
        """Returns the destabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such that along with the stabilizer
        generators above generate the full Pauli group on n qubits."""
        return [self._row_to_dense_pauli(i) for i in range(0, self.n)]

    def _measure(self, q, prng: np.random.RandomState) -> int:
        """Performs a projective measurement on the q'th qubit.

        Returns: the result (0 or 1) of the measurement.
        """
        is_commuting = True
        for i in range(self.n, 2 * self.n):
            if self.xs[i, q]:
                p = i
                is_commuting = False
                break

        if is_commuting:
            self._xs[2 * self.n, :] = False
            self._zs[2 * self.n, :] = False
            self._rs[2 * self.n] = False

            for i in range(self.n):
                if self.xs[i, q]:
                    self._rowsum(2 * self.n, self.n + i)
            return int(self._rs[2 * self.n])

        for i in range(2 * self.n):
            if i != p and self.xs[i, q]:
                self._rowsum(i, p)

        self.xs[p - self.n, :] = self.xs[p, :].copy()
        self.zs[p - self.n, :] = self.zs[p, :].copy()
        self.rs[p - self.n] = self.rs[p]

        self.xs[p, :] = False
        self.zs[p, :] = False

        self.zs[p, q] = True

        self.rs[p] = bool(prng.randint(2))

        return int(self.rs[p])
