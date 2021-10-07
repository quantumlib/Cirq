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
import fractions
from typing import (
    AbstractSet,
    Any,
    cast,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import abc

import math
import numpy as np
import sympy

from cirq import value, protocols
from cirq.ops import raw_types
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

TSelf = TypeVar('TSelf', bound='EigenGate')


EigenComponent = NamedTuple(
    'EigenComponent',
    [
        # The θ in λ = exp(i π θ) where λ is a unique eigenvalue. The exponent
        # factor is used, instead of just a raw unit complex number, because it
        # disambiguates several cases. For example, when λ=-1 you can set θ to
        # -1 instead of +1 resulting in square root operations returning -i
        # instead of +i.
        ('eigenvalue_exponent_factor', float),
        # The projection matrix onto the eigenspace of the eigenvalue. Must
        # equal Σ_k |λ_k⟩⟨λ_k| where the |λ_k⟩ vectors form an orthonormal
        # basis for the eigenspace.
        ('eigenspace_projector', np.ndarray),
    ],
)


@value.value_equality(distinct_child_types=True, approximate=True)
class EigenGate(raw_types.Gate):
    """A gate with a known eigendecomposition.

    EigenGate is particularly useful when one wishes for different parts of
    the same eigenspace to be extrapolated differently. For example, if a gate
    has a 2-dimensional eigenspace with eigenvalue -1, but one wishes for the
    square root of the gate to split this eigenspace into a part with
    eigenvalue i and a part with eigenvalue -i, then EigenGate allows this
    functionality to be unambiguously specified via the _eigen_components
    method.
    """

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def __init__(
        self, *, exponent: value.TParamVal = 1.0, global_shift: float = 0.0  # Forces keyword args.
    ) -> None:
        """Initializes the parameters used to compute the gate's matrix.

        The eigenvalue of each eigenspace of a gate is computed by

        1. Starting with an angle in half turns as returned by the gate's
        ``_eigen_components`` method:

                θ

        2. Shifting the angle by `global_shift`:

                θ + s

        3. Scaling the angle by `exponent`:

                (θ + s) * e

        4. Converting from half turns to a complex number on the unit circle:

                exp(i * pi * (θ + s) * e)

        Args:
            exponent: The t in gate**t. Determines how much the eigenvalues of
                the gate are scaled by. For example, eigenvectors phased by -1
                when `gate**1` is applied will gain a relative phase of
                e^{i pi exponent} when `gate**exponent` is applied (relative to
                eigenvectors unaffected by `gate**1`).
            global_shift: Offsets the eigenvalues of the gate at exponent=1.
                In effect, this controls a global phase factor on the gate's
                unitary matrix. The factor is:

                    exp(i * pi * global_shift * exponent)

                For example, `cirq.X**t` uses a `global_shift` of 0 but
                `cirq.rx(t)` uses a `global_shift` of -0.5, which is why
                `cirq.unitary(cirq.rx(pi))` equals -iX instead of X.
        """
        if isinstance(exponent, complex):
            if exponent.imag:
                raise ValueError(f"Gate exponent must be real. Invalid Value: {exponent}")
            exponent = exponent.real
        self._exponent = exponent
        self._global_shift = global_shift
        self._canonical_exponent_cached = None

    # pylint: enable=missing-raises-doc
    @property
    def exponent(self) -> value.TParamVal:
        return self._exponent

    @property
    def global_shift(self) -> float:
        return self._global_shift

    # virtual method
    def _with_exponent(self: TSelf, exponent: value.TParamVal) -> 'EigenGate':
        """Return the same kind of gate, but with a different exponent.

        Child classes should override this method if they have an __init__
        method with a differing signature.
        """
        # pylint: disable=unexpected-keyword-arg
        if self._global_shift == 0:
            return type(self)(exponent=exponent)
        return type(self)(exponent=exponent, global_shift=self._global_shift)
        # pylint: enable=unexpected-keyword-arg

    def _diagram_exponent(
        self, args: 'protocols.CircuitDiagramInfoArgs', *, ignore_global_phase: bool = True
    ):
        """The exponent to use in circuit diagrams.

        Basically, this just canonicalizes the exponent in a way that is
        insensitive to global phase. Only relative phases affect the "true"
        exponent period, and since we omit global phase detail in diagrams this
        is the appropriate canonicalization to use. To use the absolute period
        instead of the relative period (e.g. for when printing Rx(rads) style
        symbols, where rads=pi and rads=-pi are equivalent but should produce
        different text) set 'ignore_global_phase' to False.

        Note that the exponent is canonicalized into the range
            (-period/2, period/2]
        and that this canonicalization happens after rounding, so that e.g.
        X^-0.999999 shows as X instead of X^-1 when using a digit precision of
        3.

        Args:
            args: The diagram args being used to produce the diagram.
            ignore_global_phase: Determines whether the global phase of the
                operation is considered when computing the period of the
                exponent.

        Returns:
            A rounded canonicalized exponent.
        """
        if not isinstance(self._exponent, (int, float)):
            return self._exponent
        result = float(self._exponent)

        if ignore_global_phase:
            # Compute global-phase-independent period of the gate.
            shifts = list(self._eigen_shifts())
            relative_shifts = {e - shifts[0] for e in shifts[1:]}
            relative_periods = [abs(2 / e) for e in relative_shifts if e != 0]
            diagram_period = _approximate_common_period(relative_periods)
        else:
            # Use normal period of the gate.
            diagram_period = self._period()
        if diagram_period is None:
            return result

        # Canonicalize the rounded exponent into (-period/2, period/2].
        if args.precision is not None:
            result = np.around(result, args.precision).item()
        h = diagram_period / 2
        if not (-h < result <= h):
            result = h - result
            result %= diagram_period
            result = h - result

        return result

    def _format_exponent_as_angle(
        self,
        args: 'protocols.CircuitDiagramInfoArgs',
        order: int = 2,
    ) -> str:
        """Returns string with exponent expressed as angle in radians.

        Args:
            args: CircuitDiagramInfoArgs describing the desired drawing style.
            order: Exponent corresponding to full rotation by 2π.

        Returns:
            Angle in radians corresponding to the exponent of self and
            formatted according to style described by args.
        """
        exponent = self._diagram_exponent(args, ignore_global_phase=False)
        pi = sympy.pi if protocols.is_parameterized(exponent) else np.pi
        return args.format_radians(radians=2 * pi * exponent / order)

    # virtual method
    def _eigen_shifts(self) -> List[float]:
        """Describes the eigenvalues of the gate's matrix.

        By default, this just extracts the shifts by calling
        self._eigen_components(). However, because that method generates
        matrices it may be extremely expensive.

        Returns:
            A list of floats. Each float in the list corresponds to one of the
            eigenvalues of the gate's matrix, before accounting for any global
            shift. Each float is the θ in λ = exp(i π θ) (where λ is the
            eigenvalue).
        """
        return [e[0] for e in self._eigen_components()]

    @abc.abstractmethod
    def _eigen_components(self) -> List[Union[EigenComponent, Tuple[float, np.ndarray]]]:
        """Describes the eigendecomposition of the gate's matrix.

        Returns:
            A list of EigenComponent tuples. Each tuple in the list
            corresponds to one of the eigenspaces of the gate's matrix. Each
            tuple has two elements. The first element of a tuple is the θ in
            λ = exp(i π θ) (where λ is the eigenvalue of the eigenspace). The
            second element is a projection matrix onto the eigenspace.

        Examples:
            The Pauli Z gate's eigencomponents are:

                [
                    (0, np.array([[1, 0],
                                  [0, 0]])),
                    (1, np.array([[0, 0],
                                  [0, 1]])),
                ]

            Valid eigencomponents for Rz(π) = -iZ are:

                [
                    (-0.5, np.array([[1, 0],
                                    [0, 0]])),
                    (+0.5, np.array([[0, 0],
                                     [0, 1]])),
                ]

            But in principle you could also use this:

                [
                    (+1.5, np.array([[1, 0],
                                    [0, 0]])),
                    (-0.5, np.array([[0, 0],
                                     [0, 1]])),
                ]

                The choice between -0.5 and +1.5 does not affect the gate's
                matrix, but it does affect the matrix of powers of the gates
                (because (x+2)*s != x*s (mod 2) when s is a real number).

            The Pauli X gate's eigencomponents are:

                [
                    (0, np.array([[0.5, 0.5],
                                  [0.5, 0.5]])),
                    (1, np.array([[+0.5, -0.5],
                                  [-0.5, +0.5]])),
                ]
        """

    def _period(self) -> Optional[float]:
        """Determines how the exponent parameter is canonicalized when equating.

        Returns:
            None if the exponent should not be canonicalized. Otherwise a float
            indicating the period of the exponent. If the period is p, then a
            given exponent will be shifted by p until it is in the range
            (-p/2, p/2] during initialization.
        """
        exponents = {e + self._global_shift for e in self._eigen_shifts()}
        real_periods = [abs(2 / e) for e in exponents if e != 0]
        return _approximate_common_period(real_periods)

    def __pow__(self: TSelf, exponent: Union[float, sympy.Symbol]) -> 'EigenGate':
        new_exponent = protocols.mul(self._exponent, exponent, NotImplemented)
        if new_exponent is NotImplemented:
            return NotImplemented
        return self._with_exponent(exponent=new_exponent)

    @property
    def _canonical_exponent(self):
        if self._canonical_exponent_cached is None:
            period = self._period()
            if not period or protocols.is_parameterized(self._exponent):
                self._canonical_exponent_cached = self._exponent
            else:
                self._canonical_exponent_cached = self._exponent % period
        return self._canonical_exponent_cached

    def _value_equality_values_(self):
        return self._canonical_exponent, self._global_shift

    def _value_equality_approximate_values_(self):
        period = self._period()
        if not period or protocols.is_parameterized(self._exponent):
            exponent = self._exponent
        else:
            exponent = value.PeriodicValue(self._exponent, period)
        return exponent, self._global_shift

    def _trace_distance_bound_(self) -> Optional[float]:
        if protocols.is_parameterized(self._exponent):
            return None
        angles = np.pi * (np.array(self._eigen_shifts()) * self._exponent % 2)
        return protocols.trace_distance_from_angle_list(angles)

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        if self._is_parameterized_():
            return NotImplemented
        e = cast(float, self._exponent)
        return np.sum(
            [
                component * 1j ** (2 * e * (half_turns + self._global_shift))
                for half_turns, component in self._eigen_components()
            ],
            axis=0,
        )

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._exponent)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._exponent)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'EigenGate':
        return self._with_exponent(exponent=resolver.value_of(self._exponent, recursive))

    def _equal_up_to_global_phase_(self, other, atol):
        if not isinstance(other, EigenGate):
            return NotImplemented

        exponents = (self.exponent, other.exponent)
        exponents_is_parameterized = tuple(protocols.is_parameterized(e) for e in exponents)
        if all(exponents_is_parameterized) and exponents[0] != exponents[1]:
            return False
        if any(exponents_is_parameterized):
            return False
        self_without_phase = self._with_exponent(self.exponent)
        self_without_phase._global_shift = 0
        self_without_exp_or_phase = self_without_phase._with_exponent(0)
        self_without_exp_or_phase._global_shift = 0
        other_without_phase = other._with_exponent(other.exponent)
        other_without_phase._global_shift = 0
        other_without_exp_or_phase = other_without_phase._with_exponent(0)
        other_without_exp_or_phase._global_shift = 0
        if not protocols.approx_eq(
            self_without_exp_or_phase, other_without_exp_or_phase, atol=atol
        ):
            return False

        period = self_without_phase._period()
        canonical_diff = (exponents[0] - exponents[1]) % period
        return np.isclose(canonical_diff, 0, atol=atol)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['exponent', 'global_shift'])


def _lcm(vals: Iterable[int]) -> int:
    t = 1
    for r in vals:
        t = t * r // math.gcd(t, r)
    return t


def _approximate_common_period(
    periods: List[float], approx_denom: int = 60, reject_atol: float = 1e-8
) -> Optional[float]:
    """Finds a value that is nearly an integer multiple of multiple periods.

    The returned value should be the smallest non-negative number with this
    property. If `approx_denom` is too small the computation can fail to satisfy
    the `reject_atol` criteria and return `None`. This is actually desirable
    behavior, since otherwise the code would e.g. return a nonsense value when
    asked to compute the common period of `np.e` and `np.pi`.

    Args:
        periods: The result must be an approximate integer multiple of each of
            these.
        approx_denom: Determines how the floating point values are rounded into
            rational values (so that integer methods such as lcm can be used).
            Each floating point value f_k will be rounded to a rational number
            of the form n_k / approx_denom. If you want to recognize rational
            periods of the form i/d then d should divide `approx_denom`.
        reject_atol: If the computed approximate common period is at least this
            far from an integer multiple of any of the given periods, then it
            is discarded and `None` is returned instead.

    Returns:
        The approximate common period, or else `None` if the given
        `approx_denom` wasn't sufficient to approximate the common period to
        within the given `reject_atol`.
    """
    if not periods:
        return None
    if any(e == 0 for e in periods):
        return None
    if len(periods) == 1:
        return abs(periods[0])
    approx_rational_periods = [
        fractions.Fraction(int(np.round(abs(p) * approx_denom)), approx_denom) for p in periods
    ]
    common = float(_common_rational_period(approx_rational_periods))

    for p in periods:
        if p != 0 and abs(p * np.round(common / p) - common) > reject_atol:
            return None

    return common


def _common_rational_period(rational_periods: List[fractions.Fraction]) -> fractions.Fraction:
    """Finds the least common integer multiple of some fractions.

    The solution is the smallest positive integer c such that there
    exists integers n_k satisfying p_k * n_k = c for all k.
    """
    assert rational_periods, "no well-defined solution for an empty list"
    common_denom = _lcm(p.denominator for p in rational_periods)
    int_periods = [p.numerator * common_denom // p.denominator for p in rational_periods]
    int_common_period = _lcm(int_periods)
    return fractions.Fraction(int_common_period, common_denom)
