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

from typing import Any, Dict, List, Sequence, Union
import numpy as np

import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state


@value.value_equality
class StabilizerStateChForm(qis.StabilizerState):
    r"""A representation of stabilizer states using the CH form,

        $|\psi> = \omega U_C U_H |s>$

    This representation keeps track of overall phase.

    Reference: https://arxiv.org/abs/1808.00128
    """

    def __init__(self, num_qubits: int, initial_state: int = 0) -> None:
        """Initializes StabilizerStateChForm
        Args:
            num_qubits: The number of qubits in the system.
            initial_state: The computational basis representation of the
                state as a big endian int.
        """
        self.n = num_qubits

        # The state is represented by a set of binary matrices and vectors.
        # See Section IVa of Bravyi et al
        self.G = np.eye(self.n, dtype=bool)

        self.F = np.eye(self.n, dtype=bool)
        self.M = np.zeros((self.n, self.n), dtype=bool)
        self.gamma = np.zeros(self.n, dtype=int)

        self.v = np.zeros(self.n, dtype=bool)
        self.s = np.zeros(self.n, dtype=bool)

        self.omega: complex = 1

        # Apply X for every non-zero element of initial_state
        for i, val in enumerate(
            big_endian_int_to_digits(initial_state, digit_count=num_qubits, base=2)
        ):
            if val:
                self.apply_x(i)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['n', 'G', 'F', 'M', 'gamma', 'v', 's', 'omega'])

    @classmethod
    def _from_json_dict_(cls, n, G, F, M, gamma, v, s, omega, **kwargs):
        copy = StabilizerStateChForm(n)

        copy.G = np.array(G)
        copy.F = np.array(F)
        copy.M = np.array(M)
        copy.gamma = np.array(gamma)
        copy.v = np.array(v)
        copy.s = np.array(s)
        copy.omega = omega

        return copy

    def _value_equality_values_(self) -> Any:
        return (self.n, self.G, self.F, self.M, self.gamma, self.v, self.s, self.omega)

    def copy(self, deep_copy_buffers: bool = True) -> 'cirq.StabilizerStateChForm':
        copy = StabilizerStateChForm(self.n)

        copy.G = self.G.copy()
        copy.F = self.F.copy()
        copy.M = self.M.copy()
        copy.gamma = self.gamma.copy()
        copy.v = self.v.copy()
        copy.s = self.s.copy()
        copy.omega = self.omega

        return copy

    def __str__(self) -> str:
        """Return the state vector string representation of the state."""
        return cirq.dirac_notation(self.to_state_vector())

    def __repr__(self) -> str:
        """Return the CH form representation of the state."""
        return f'StabilizerStateChForm(num_qubits={self.n!r})'

    def inner_product_of_state_and_x(self, x: int) -> Union[float, complex]:
        """Returns the amplitude of x'th element of
        the state vector, i.e. <x|psi>"""
        if type(x) == int:
            y = cirq.big_endian_int_to_bits(x, bit_count=self.n)

        mu = sum(y * self.gamma)

        u = np.zeros(self.n, dtype=bool)
        for p in range(self.n):
            if y[p]:
                u ^= self.F[p, :]
                mu += 2 * (sum(self.M[p, :] & u) % 2)
        return (
            self.omega
            * 2 ** (-sum(self.v) / 2)
            * 1j**mu
            * (-1) ** sum(self.v & u & self.s)
            * bool(np.all(self.v | (u == self.s)))
        )

    def state_vector(self) -> np.ndarray:
        wf = np.zeros(2**self.n, dtype=complex)

        for x in range(2**self.n):
            wf[x] = self.inner_product_of_state_and_x(x)

        return wf

    def _S_right(self, q):
        r"""Right multiplication version of S gate."""
        self.M[:, q] ^= self.F[:, q]
        self.gamma[:] = (self.gamma[:] - self.F[:, q]) % 4

    def _CZ_right(self, q, r):
        r"""Right multiplication version of CZ gate."""
        self.M[:, q] ^= self.F[:, r]
        self.M[:, r] ^= self.F[:, q]
        self.gamma[:] = (self.gamma[:] + 2 * self.F[:, q] * self.F[:, r]) % 4

    def _CNOT_right(self, q, r):
        r"""Right multiplication version of CNOT gate."""
        self.G[:, q] ^= self.G[:, r]
        self.F[:, r] ^= self.F[:, q]
        self.M[:, q] ^= self.M[:, r]

    def update_sum(self, t, u, delta=0, alpha=0):
        """Implements the transformation (Proposition 4 in Bravyi et al)

        ``i^alpha U_H (|t> + i^delta |u>) = omega W_C W_H |s'>``
        """
        if np.all(t == u):
            self.s = t
            self.omega *= 1 / np.sqrt(2) * (-1) ** alpha * (1 + 1j**delta)
            return
        set0 = np.where((~self.v) & (t ^ u))[0]
        set1 = np.where(self.v & (t ^ u))[0]

        # implement Vc
        if len(set0) > 0:
            q = set0[0]
            for i in set0:
                if i != q:
                    self._CNOT_right(q, i)
            for i in set1:
                self._CZ_right(q, i)
        elif len(set1) > 0:
            q = set1[0]
            for i in set1:
                if i != q:
                    self._CNOT_right(i, q)

        e = np.zeros(self.n, dtype=bool)
        e[q] = True

        if t[q]:
            y = u ^ e
            z = u
        else:
            y = t
            z = t ^ e

        (omega, a, b, c) = self._H_decompose(self.v[q], y[q], z[q], delta)

        self.s = y
        self.s[q] = c
        self.omega *= (-1) ** alpha * omega

        if a:
            self._S_right(q)
        self.v[q] ^= b ^ self.v[q]

    def _H_decompose(self, v, y, z, delta):
        """Determines the transformation

                H^v (|y> + i^delta |z>) = omega S^a H^b |c>

        where the state represents a single qubit.

        Input: v,y,z are boolean; delta is an integer (mod 4)
        Outputs: a,b,c are boolean; omega is a complex number

        Precondition: y != z"""
        if y == z:
            raise ValueError('|y> is equal to |z>')

        if not v:
            omega = (1j) ** (delta * int(y))

            delta2 = ((-1) ** y * delta) % 4
            c = bool((delta2 >> 1))
            a = bool(delta2 & 1)
            b = True
        else:
            if not (delta & 1):
                a = False
                b = False
                c = bool(delta >> 1)
                omega = (-1) ** (c & y)
            else:
                omega = 1 / np.sqrt(2) * (1 + 1j**delta)
                b = True
                a = True
                c = not ((delta >> 1) ^ y)

        return omega, a, b, c

    def to_state_vector(self) -> np.ndarray:
        arr = np.zeros(2**self.n, dtype=complex)

        for x in range(len(arr)):
            arr[x] = self.inner_product_of_state_and_x(x)

        return arr

    def _measure(self, q, prng: np.random.RandomState) -> int:
        """Measures the q'th qubit.

        Reference: Section 4.1 "Simulating measurements"

        Returns: Computational basis measurement as 0 or 1.
        """
        w = self.s.copy()
        for i, v_i in enumerate(self.v):
            if v_i == 1:
                w[i] = bool(prng.randint(2))
        x_i = sum(w & self.G[q, :]) % 2
        # Project the state to the above measurement outcome.
        self.project_Z(q, x_i)
        return x_i

    def project_Z(self, q, z):
        """Applies a Z projector on the q'th qubit.

        Returns: a normalized state with Z_q |psi> = z |psi>
        """
        t = self.s.copy()
        u = (self.G[q, :] & self.v) ^ self.s
        delta = (2 * sum((self.G[q, :] & (~self.v)) & self.s) + 2 * z) % 4

        if np.all(t == u):
            self.omega /= np.sqrt(2)

        self.update_sum(t, u, delta=delta)

    def kron(self, other: 'cirq.StabilizerStateChForm') -> 'cirq.StabilizerStateChForm':
        n = self.n + other.n
        copy = StabilizerStateChForm(n)
        copy.G[: self.n, : self.n] = self.G
        copy.G[self.n :, self.n :] = other.G
        copy.F[: self.n, : self.n] = self.F
        copy.F[self.n :, self.n :] = other.F
        copy.M[: self.n, : self.n] = self.M
        copy.M[self.n :, self.n :] = other.M
        copy.gamma = np.concatenate([self.gamma, other.gamma])
        copy.v = np.concatenate([self.v, other.v])
        copy.s = np.concatenate([self.s, other.s])
        copy.omega = self.omega * other.omega
        return copy

    def reindex(self, axes: Sequence[int]) -> 'cirq.StabilizerStateChForm':
        copy = StabilizerStateChForm(self.n)
        copy.G = self.G[axes][:, axes]
        copy.F = self.F[axes][:, axes]
        copy.M = self.M[axes][:, axes]
        copy.gamma = self.gamma[axes]
        copy.v = self.v[axes]
        copy.s = self.s[axes]
        copy.omega = self.omega
        return copy

    def apply_x(self, axis: int, exponent: float = 1, global_shift: float = 0):
        if exponent % 2 != 0:
            if exponent % 0.5 != 0.0:
                raise ValueError('X exponent must be half integer')  # pragma: no cover
            self.apply_h(axis)
            self.apply_z(axis, exponent)
            self.apply_h(axis)
        self.omega *= _phase(exponent, global_shift)

    def apply_y(self, axis: int, exponent: float = 1, global_shift: float = 0):
        if exponent % 0.5 != 0.0:
            raise ValueError('Y exponent must be half integer')  # pragma: no cover
        shift = _phase(exponent, global_shift)
        if exponent % 2 == 0:
            self.omega *= shift
        elif exponent % 2 == 0.5:
            self.apply_z(axis)
            self.apply_h(axis)
            self.omega *= shift * (1 + 1j) / (2**0.5)
        elif exponent % 2 == 1:
            self.apply_z(axis)
            self.apply_h(axis)
            self.apply_z(axis)
            self.apply_h(axis)
            self.omega *= shift * 1j
        elif exponent % 2 == 1.5:
            self.apply_h(axis)
            self.apply_z(axis)
            self.omega *= shift * (1 - 1j) / (2**0.5)

    def apply_z(self, axis: int, exponent: float = 1, global_shift: float = 0):
        if exponent % 2 != 0:
            if exponent % 0.5 != 0.0:
                raise ValueError('Z exponent must be half integer')  # pragma: no cover
            effective_exponent = exponent % 2
            for _ in range(int(effective_exponent * 2)):
                # Prescription for S left multiplication.
                # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
                self.M[axis, :] ^= self.G[axis, :]
                self.gamma[axis] = (self.gamma[axis] - 1) % 4
        self.omega *= _phase(exponent, global_shift)

    def apply_h(self, axis: int, exponent: float = 1, global_shift: float = 0):
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('H exponent must be integer')  # pragma: no cover
            # Prescription for H left multiplication
            # Reference: https://arxiv.org/abs/1808.00128
            # Equations 48, 49 and Proposition 4
            t = self.s ^ (self.G[axis, :] & self.v)
            u = self.s ^ (self.F[axis, :] & (~self.v)) ^ (self.M[axis, :] & self.v)
            alpha = sum(self.G[axis, :] & (~self.v) & self.s) % 2
            beta = sum(self.M[axis, :] & (~self.v) & self.s)
            beta += sum(self.F[axis, :] & self.v & self.M[axis, :])
            beta += sum(self.F[axis, :] & self.v & self.s)
            beta %= 2
            delta = (self.gamma[axis] + 2 * (alpha + beta)) % 4
            self.update_sum(t, u, delta=delta, alpha=alpha)
        self.omega *= _phase(exponent, global_shift)

    def apply_cz(
        self, control_axis: int, target_axis: int, exponent: float = 1, global_shift: float = 0
    ):
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('CZ exponent must be integer')  # pragma: no cover
            # Prescription for CZ left multiplication.
            # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
            self.M[control_axis, :] ^= self.G[target_axis, :]
            self.M[target_axis, :] ^= self.G[control_axis, :]
        self.omega *= _phase(exponent, global_shift)

    def apply_cx(
        self, control_axis: int, target_axis: int, exponent: float = 1, global_shift: float = 0
    ):
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('CX exponent must be integer')  # pragma: no cover
            # Prescription for CX left multiplication.
            # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
            self.gamma[control_axis] = (
                self.gamma[control_axis]
                + self.gamma[target_axis]
                + 2 * (sum(self.M[control_axis, :] & self.F[target_axis, :]) % 2)
            ) % 4
            self.G[target_axis, :] ^= self.G[control_axis, :]
            self.F[control_axis, :] ^= self.F[target_axis, :]
            self.M[control_axis, :] ^= self.M[target_axis, :]
        self.omega *= _phase(exponent, global_shift)

    def apply_global_phase(self, coefficient: value.Scalar):
        self.omega *= coefficient

    def measure(
        self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
    ) -> List[int]:
        return [self._measure(axis, random_state.parse_random_state(seed)) for axis in axes]


def _phase(exponent, global_shift):
    return np.exp(1j * np.pi * global_shift * exponent)
