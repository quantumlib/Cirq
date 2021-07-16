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

        # The following computation is based on Theorem 36 in
        # https://arxiv.org/pdf/2009.03218.pdf.
        # Any pauli string (one stabilizer) in Clifford Tableau should be able to be expressed as
        #    (1i)^p (-1)^s X^(mx) Z^(mz)
        # where p and s are binary scalar and mx and mz are binary vectors.
        num_ys1 = np.sum(m1[:, : self.n] * m1[:, self.n :], axis=1)
        num_ys2 = np.sum(m2[:, : self.n] * m2[:, self.n :], axis=1)

        p1 = np.mod(num_ys1, 2)
        p2 = np.mod(num_ys2, 2)

        # Note the `s` is not equal to `r`, which depends on the number of Y gates.
        # For example, r * Y_1Y_2Y_3 can be expanded into i^3 * r * X_1Z_1 X_2Z_2 X_3Z_3.
        # The global phase is i * (-1) * r ==> s = r + 1 and p = 1.
        s1 = self.rs.astype(int) + np.mod(num_ys1, 4) // 2
        s2 = second.rs.astype(int) + np.mod(num_ys2, 4) // 2

        lmbda = np.zeros((2 * self.n, 2 * self.n))
        lmbda[: self.n, self.n :] = np.eye(self.n)

        m_12 = np.mod(m1 @ m2, 2)
        p_12 = np.mod(p1 + m1 @ p2, 2)
        s_12 = (
            s1
            + m1 @ s2
            + p1 * (m1 @ p2)
            + np.diag(m1 @ np.tril(np.outer(p2, p2.T) + m2 @ lmbda @ m2.T, -1) @ m1.T)
        )
        num_ys12 = np.sum(m_12[:, : self.n] * m_12[:, self.n :], axis=1)
        merged_sign = np.mod(p_12 + 2 * s_12 - num_ys12, 4) // 2

        merged_tableau = CliffordTableau(num_qubits=self.n)
        merged_tableau.xs = m_12[:, : self.n]
        merged_tableau.zs = m_12[:, self.n :]
        merged_tableau.rs = merged_sign

        return merged_tableau

    def inverse(self) -> 'CliffordTableau':
        """Returns the inverse Clifford tableau of this tableau."""
        ret_table = CliffordTableau(num_qubits=self.n)
        # It relies on the symplectic property of Clifford tableau.
        #  [A^T C^T  [0 I  [A B     [0 I
        #   B^T D^T]  I 0]  C D]  =  I 0]
        # So the inverse is [[D^T B^T], [C^T A^T]]
        ret_table.xs[: self.n] = self.zs[self.n :].T
        ret_table.zs[: self.n] = self.zs[: self.n].T
        ret_table.xs[self.n :] = self.xs[self.n :].T
        ret_table.zs[self.n :] = self.xs[: self.n].T

        # Update the sign -- rs.
        # The idea is noting the sign of tabluea `a` contributes to the composed tableau
        # `a.then(b)` directly. (While the sign in `b` need take very complicated transformation.)
        # Refer above `then` function implementation for more details.
        ret_table.rs = ret_table.then(self).rs
        return ret_table

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

    # TODO(#3388) Add summary line to docstring.
    # pylint: disable=docstring-first-line-empty
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

    # pylint: enable=docstring-first-line-empty
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
