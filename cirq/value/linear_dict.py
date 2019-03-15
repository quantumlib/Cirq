# Copyright 2018 The Cirq Developers
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

"""Linear combination represented as mapping of things to coefficients."""

from typing import Any, Dict, Mapping, Union

Scalar = Union[complex, float]


class LinearDict(Dict[Any, Scalar]):
    """Represents linear combination of things.

    LinearDict implements the basic linear algebraic operations of vector
    addition and scalar multiplication for linear combinations of abstract
    vectors. Keys represent the vectors, values represent their coefficients.
    The only requirement on the keys is that they be hashable.

    A consequence of treating keys as opaque is that all relationships between
    the keys other than equality are ignored. In particular, keys are allowed
    to be linearly dependent.
    """
    def __init__(self, terms: Mapping[Any, Scalar]) -> None:
        """Initializes linear combination from a collection of terms.

        Args:
            terms: Mapping of abstract vectors to coefficients in the linear
                combination being initialized.
        """
        super().__init__()
        self.update(terms)

    def clean(self, *, atol: float=1e-9) -> None:
        """Modifies self by deleting terms with small coefficients."""
        negligible = [v for v, c in self.items() if abs(c) <= atol]
        for v in negligible:
            del self[v]

    def copy(self) -> 'LinearDict':
        return LinearDict(super().copy())

    def __getitem__(self, vector: Any) -> Scalar:
        return super().get(vector, 0)

    def __iadd__(self, other: 'LinearDict') -> 'LinearDict':
        for vector, other_coefficient in other.items():
            old_coefficient = super().get(vector, 0)
            new_coefficient = old_coefficient + other_coefficient
            super().__setitem__(vector, new_coefficient)
        self.clean(atol=0)
        return self

    def __add__(self, other: 'LinearDict') -> 'LinearDict':
        result = self.copy()
        result += other
        return result

    def __isub__(self, other: 'LinearDict') -> 'LinearDict':
        for vector, other_coefficient in other.items():
            old_coefficient = super().get(vector, 0)
            new_coefficient = old_coefficient - other_coefficient
            super().__setitem__(vector, new_coefficient)
        self.clean(atol=0)
        return self

    def __sub__(self, other: 'LinearDict') -> 'LinearDict':
        result = self.copy()
        result -= other
        return result

    def __neg__(self) -> 'LinearDict':
        return LinearDict({v: -c for v, c in self.items()})

    def __imul__(self, a: Scalar) -> 'LinearDict':
        for vector in self:
            self[vector] *= a
        self.clean(atol=0)
        return self

    def __mul__(self, a: Scalar) -> 'LinearDict':
        result = self.copy()
        result *= a
        return result

    def __rmul__(self, a: Scalar) -> 'LinearDict':
        return self.__mul__(a)

    def __truediv__(self, a: Scalar) -> 'LinearDict':
        return self.__mul__(1 / a)

    def __bool__(self) -> bool:
        return not all(c == 0 for c in self.values())

    def __eq__(self, other: Any) -> bool:
        """Checks whether two linear combinations are exactly equal.

        Presence or absence of terms with coefficients exactly equal to
        zero does not affect outcome.

        Not appropriate for most practical purposes due to sensitivity to
        numerical error in floating point coefficients. Use cirq.approx_eq()
        instead.
        """
        if not isinstance(other, LinearDict):
            return NotImplemented

        all_vs = set(self.keys()) | set(other.keys())
        return all(self[v] == other[v] for v in all_vs)

    def __ne__(self, other: Any) -> bool:
        """Checks whether two linear combinations are not exactly equal.

        See __eq__().
        """
        return not self == other

    def _approx_eq_(self, other: Any, atol: float) -> bool:
        """Checks whether two linear combinations are approximately equal."""
        if not isinstance(other, LinearDict):
            return NotImplemented

        all_vs = set(self.keys()) | set(other.keys())
        return all(abs(self[v] - other[v]) < atol for v in all_vs)

    def __repr__(self) -> str:
        coefficients = dict(self)
        return 'cirq.LinearDict({!r})'.format(coefficients)

    @staticmethod
    def _term_to_str(vector: Any, coefficient: Scalar) -> str:
        coefficient = complex(coefficient)
        if abs(coefficient.real) < 1e-4 and abs(coefficient.imag) < 1e-4:
            return ''
        if abs(coefficient.real) < 1e-4:
            return '{:+.3f}i*{!s}'.format(coefficient.imag, vector)
        if abs(coefficient.imag) < 1e-4:
            return '{:+.3f}*{!s}'.format(coefficient.real, vector)
        return '+({:.3f}{:+.3f}i)*{!s}'.format(
                coefficient.real, coefficient.imag, vector)

    def __str__(self):
        s = ''.join(self._term_to_str(v, self[v]) for v in sorted(self.keys()))
        return s[1:] if s else '0'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text('LinearDict(...)')
        else:
            p.text(str(self))

