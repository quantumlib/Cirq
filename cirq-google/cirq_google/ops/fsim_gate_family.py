# Copyright 2021 The Cirq Developers
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

"""Define FSimGateFamily used to convert/accept `cirq.FSimGate` and other related gate types"""

from typing import Any, Callable, cast, Dict, Iterable, Optional, Type, TypeVar, Sequence, Union

import sympy
import numpy as np
import cirq
from cirq.ops.gateset import _gate_str

POSSIBLE_FSIM_GATES = Union[
    cirq.FSimGate,
    cirq.PhasedFSimGate,
    cirq.ISwapPowGate,
    cirq.PhasedISwapPowGate,
    cirq.CZPowGate,
    cirq.IdentityGate,
]
T = TypeVar(
    'T',
    cirq.FSimGate,
    cirq.PhasedFSimGate,
    cirq.ISwapPowGate,
    cirq.PhasedISwapPowGate,
    cirq.CZPowGate,
    cirq.IdentityGate,
)


def _exp(theta: Union[complex, sympy.Basic]):
    """Utility method to return exp(theta) using numpy or sympy, depending on the type of theta."""
    return sympy.exp(theta) if cirq.is_parameterized(theta) else np.exp(theta)


def _gates_to_str(gates: Iterable[Any], gettr: Callable[[Any], str] = _gate_str) -> str:
    """Converts a list of gates (types/instances) to string by calling gettr (str/repr) on each."""
    return f'[{",".join(gettr(g) for g in gates)}]'


# Default tolerance for differences in floating point
# Note that Google protocol buffers use floats
# which trigger a conversion from double precision to single precision
# This results in errors possibly up to 1e-6
# (23 bits for mantissa in single precision)
DEFAULT_ATOL = 1e-6


class FSimGateFamily(cirq.GateFamily):
    """GateFamily useful to convert/accept `cirq.FSimGate` and other related gate types.

    This gate family is useful to work with any of the different representations of `cirq.FSimGate`
    and its related types present in Cirq. Specifically, the gate family can be used to convert or
    accept (if possible) compatible instances of any of the following `POSSIBLE_FSIM_GATES` types:

    1. `cirq.FSimGate`, `cirq.PhasedFSimGate`
    2. `cirq.ISwapPowGate`, `cirq.PhasedISwapPowGate`
    3. `cirq.CZPowGate`
    4. `cirq.IdentityGate`

    The gate family also allows an option to accept parameterized gates, assuming that the correct
    parameters would eventually be filled.

    For example,

    1. To convert to/from any of the non-parameterized `POSSIBLE_FSIM_GATES` types:
    >>> gf = cirq_google.FSimGateFamily()
    >>> assert gf.convert(cirq.FSimGate(np.pi/4, 0), cirq.ISwapPowGate) == cirq.SQRT_ISWAP_INV
    >>> assert gf.convert(cirq.PhasedFSimGate(0, 0, 0, 0, np.pi), cirq.CZPowGate) == cirq.CZ
    >>> assert gf.convert(cirq.FSimGate(-np.pi/2, sympy.Symbol("t")), cirq.ISwapPowGate) is None

    2. To convert to/from any of the parameterized `POSSIBLE_FSIM_GATES` types (assuming correct
        value of the parameter would be filled in during paramter resolution):
    >>> gf = cirq_google.FSimGateFamily(allow_symbols = True)
    >>> theta, phi = sympy.Symbol("theta"), sympy.Symbol("phi")
    >>> assert gf.convert(cirq.FSimGate(-np.pi/4, phi), cirq.ISwapPowGate) == cirq.SQRT_ISWAP
    >>> assert gf.convert(cirq.PhasedFSimGate(theta, 0, 0, 0, np.pi), cirq.CZPowGate) == cirq.CZ
    >>> assert gf.convert(cirq.FSimGate(theta, phi), cirq.IdentityGate) == cirq.IdentityGate(2)

    3. To accept instances of `gate_types_to_check` based on type / value equality check against
    gates in `gates_to_accept`, possibly accepting parameterized instances (assuming correct
    parameter value would be filled in during parameter resolution) based on `allow_symbols`:
    >>> gf = cirq_google.FSimGateFamily(
    >>>     gates_to_accept=[cirq.SQRT_ISWAP, cirq.CZPowGate, cirq_google.SYC],
    >>>     gate_types_to_check=[cirq.FSimGate],
    >>>     allow_symbols=True,
    >>> )
    >>> theta, phi = sympy.Symbol("theta"), sympy.Symbol("phi")
    >>> assert cirq.FSimGate(theta, phi) in gf # Assumes correct theta/phi will be substituted.
    >>> assert cirq_google.SYC in gf # SYC
    >>> assert cirq.FSimGate(0, np.pi / 2) in gf # CZPowGate
    >>> assert cirq.FSimGate(-np.pi / 4, phi) in gf # SQRT_ISWAP
    >>> assert cirq.FSimGate(-np.pi / 8, phi) not in gf # No value of `phi` would make it equal to
    >>>                                                 # any gate/gate type in `gates_to_accept`.
    >>> assert cirq.CZ ** 0.25 not in gf # CZPowGate not in gate_types_to_check
    >>> assert cirq.SQRT_ISWAP not in gf # ISwapPowGate not in gate_types_to_check
    """

    def __init__(
        self,
        *,
        gates_to_accept: Sequence[Union[Type[POSSIBLE_FSIM_GATES], POSSIBLE_FSIM_GATES]] = (),
        gate_types_to_check: Sequence[Type[POSSIBLE_FSIM_GATES]] = (),
        allow_symbols: bool = False,
        atol=DEFAULT_ATOL,
    ):
        """Inits `cirq_google.FSimGateFamily`.

        Args:
            gates_to_accept: List of gate types or instances to be accepted. All elements
                should be either types from `POSSIBLE_FSIM_GATES` or non-parameterized
                instances of those types.
            gate_types_to_check: List of `POSSIBLE_FSIM_GATES` types whose instances should be
                considered when trying to match against gates present in `gates_to_accept`.
                Defaults to all `POSSIBLE_FSIM_GATES` if left unspecified.
            allow_symbols: If True, both the gate conversion logic and containment predicate allow
                parameterized gate instances and return a converted gate / accept input gate if
                there exists any value of the unknown parameters which can result in a valid
                outcome.
            atol: Absolute tolerance for difference floating point comparisons.

        Raises:
            ValueError: If any gate in `gates_to_accept` is not a non-parameterized instance of /
                or a gate type from `POSSIBLE_FSIM_GATES`.
            ValueError: If any gate type in `gate_types_to_check` is not one of
                `POSSIBLE_FSIM_GATES`.
        """
        self._supported_types: Dict[
            Type[POSSIBLE_FSIM_GATES],
            Callable[[POSSIBLE_FSIM_GATES], Optional[POSSIBLE_FSIM_GATES]],
        ] = {
            cirq.FSimGate: self._convert_to_fsim,
            cirq.PhasedFSimGate: self._convert_to_phased_fsim,
            cirq.ISwapPowGate: self._convert_to_iswap,
            cirq.PhasedISwapPowGate: self._convert_to_phased_iswap,
            cirq.CZPowGate: self._convert_to_cz,
            cirq.IdentityGate: self._convert_to_identity,
        }
        if not gate_types_to_check:
            gate_types_to_check = tuple(self._supported_types.keys())

        if any(g not in self._supported_types for g in gate_types_to_check):
            raise ValueError(
                f"All gates in gate_types_to_check: {_gates_to_str(gate_types_to_check)} must "
                f"be one of {_gates_to_str(self._supported_types.keys())}."
            )

        for g in gates_to_accept:
            if isinstance(g, tuple(self._supported_types.keys())):
                if cirq.is_parameterized(g):
                    raise ValueError(
                        f"Parameterized gate {g} cannot be used in `gates_to_accept` initializer."
                    )
            elif g not in self._supported_types:
                raise ValueError(
                    f"Gate {g} in `gates_to_accept` must be either a type from or an instance of "
                    f"{_gates_to_str(self._supported_types.keys())}"
                )
        self.gates_to_accept = tuple(dict.fromkeys(gates_to_accept))
        self.gate_types_to_check = tuple(dict.fromkeys(gate_types_to_check))
        self.allow_symbols = allow_symbols
        self.atol = atol
        super().__init__(cirq.Gate)

    def _default_name(self) -> str:
        return f'FSimGateFamily: allow_symbol={self.allow_symbols}; atol={self.atol}'

    def _default_description(self) -> str:
        return (
            f'`cirq_google.FSimGateFamily` which accepts any instance of gate types in'
            f'\ngate_types_to_check: {_gates_to_str(self.gate_types_to_check)}'
            f'\nwhich matches (across types), via instance check / value equality, a gate in'
            f'\ngates_to_accept: {_gates_to_str(self.gates_to_accept)}'
        )

    def __repr__(self) -> str:
        _gate_repr = lambda x: _gate_str(x, repr)
        return (
            'cirq_google.FSimGateFamily('
            f'gates_to_accept={_gates_to_str(self.gates_to_accept, _gate_repr)}, '
            f'gate_types_to_check={_gates_to_str(self.gate_types_to_check, _gate_repr)}, '
            f'allow_symbols={self.allow_symbols}, '
            f'atol={self.atol})'
        )

    def _value_equality_values_(self) -> Any:
        return (
            frozenset(self.gates_to_accept),
            frozenset(self.gate_types_to_check),
            self.allow_symbols,
            self.atol,
        )

    def _json_dict_(self):
        accept_gates_json = [
            gate if not isinstance(gate, type) else cirq.json_cirq_type(gate)
            for gate in self.gates_to_accept
        ]
        check_gates_json = [cirq.json_cirq_type(gate) for gate in self.gate_types_to_check]
        return {
            'gates_to_accept': accept_gates_json,
            'gate_types_to_check': check_gates_json,
            'allow_symbols': self.allow_symbols,
            'atol': self.atol,
        }

    @classmethod
    def _from_json_dict_(cls, gates_to_accept, gate_types_to_check, allow_symbols, atol, **kwargs):
        accept_gates = [
            gate if not isinstance(gate, str) else cirq.cirq_type_from_json(gate)
            for gate in gates_to_accept
        ]
        check_gates = [cirq.cirq_type_from_json(gate) for gate in gate_types_to_check]
        return cls(
            gates_to_accept=accept_gates,
            gate_types_to_check=check_gates,
            allow_symbols=allow_symbols,
            atol=atol,
        )

    def _approx_eq_or_symbol(self, lhs: Any, rhs: Any) -> bool:
        lhs = lhs if isinstance(lhs, tuple) else (lhs,)
        rhs = rhs if isinstance(rhs, tuple) else (rhs,)
        assert len(lhs) == len(rhs)
        for l, r in zip(lhs, rhs):
            is_parameterized = cirq.is_parameterized(l) or cirq.is_parameterized(r)
            if (is_parameterized and not self.allow_symbols) or (
                not is_parameterized and not cirq.approx_eq(l, r, atol=self.atol)
            ):
                return False
        return True

    def _get_value_equality_values(self, g: POSSIBLE_FSIM_GATES) -> Any:
        # TODO: Remove condition once https://github.com/quantumlib/Cirq/issues/4585 is fixed.
        if type(g) == cirq.PhasedISwapPowGate:
            return (g.phase_exponent, *g._iswap._value_equality_values_())  # type: ignore
        return g._value_equality_values_()

    def _get_value_equality_values_cls(self, g: POSSIBLE_FSIM_GATES) -> Any:
        # TODO: Remove condition once https://github.com/quantumlib/Cirq/issues/4585 is fixed.
        if type(g) == cirq.PhasedISwapPowGate:
            return cirq.PhasedISwapPowGate
        return g._value_equality_values_cls_()  # type: ignore

    def _check_equal(self, g1: POSSIBLE_FSIM_GATES, g2: POSSIBLE_FSIM_GATES) -> bool:
        if not self.allow_symbols:
            return g1 == g2 and not (cirq.is_parameterized(g1) or cirq.is_parameterized(g2))
        if self._get_value_equality_values_cls(g1) != self._get_value_equality_values_cls(g2):
            return False
        return self._approx_eq_or_symbol(
            self._get_value_equality_values(g1), self._get_value_equality_values(g2)
        )

    def _predicate(self, gate: cirq.Gate) -> bool:
        """Checks whether `cirq.Gate` instance `gate` belongs to this GateFamily.

        To get accepted, `gate` must be an instance of a type present in `self.gate_types_to_check`
        and match a gate / gate type present in `self.gates_to_accept`.

        Let `target_gate` be an element of `self.gates_to_accept`. `gate` would match `target_gate`
        if
            a) Type Equality Check: `target_gate` is one of `POSSIBLE_FSIM_GATES` types and `gate`
               can be converted to an instance of `target_gate`.
            b) Value Equality Check: `target_gate` is an instance of `POSSIBLE_FSIM_GATES` and is
               equal to `gate` (modulo type conversion). Note that value equality for parameterized
               gates tries to be lenient and assumes that the correct parameters would eventually
               be filled during parameter resolution.
        Args:
           gate: `cirq.Gate` instance which should be checked for containment.
        """
        if not isinstance(gate, self.gate_types_to_check):
            return False
        gate = cast(POSSIBLE_FSIM_GATES, gate)
        for g in self.gates_to_accept:
            if isinstance(g, type):
                cg = self.convert(gate, cast(type, g))  # mypy hack.
                if cg is not None:
                    return True
            elif isinstance(g, cirq.Gate):
                for target in type(gate).mro():
                    if target in self.gate_types_to_check:
                        cg = self.convert(g, target)
                        if cg is None:
                            continue
                        if self._check_equal(gate, cg):
                            return True
                        break
        return False

    def convert(self, gate: cirq.Gate, target_gate_type: Type[T]) -> Optional[T]:
        """Converts, if possible, the given `gate` to an equivalent instance of `target_gate_type`.

        This method can be used for converting instances of `POSSIBLE_FSIM_GATES` to other
        equivalent types from the same group. For example, you can convert a sqrt iswap gate
        to an equivalent fsim gate by calling:

        >>> gf = cirq_google.FSimGateFamily()
        >>> assert gf.convert(cirq.SQRT_ISWAP, cirq.FSimGate) == cirq.FSimGate(-np.pi/4, 0)

        The method can also be used for converting parameterized gate instances, by setting
        `allow_symbols=True` in the gate family constructor. Note that, conversion of
        paramaterized gate instances tries to be lenient and assumes that the correct
        parameters would eventually be filled during parameter resolution. This can also result
        in dropping extra parameters during type conversion, assuming the dropped parameters
        would be supplied the correct values. For example:

        >>> gf = cirq_google.FSimGateFamily(allow_symbols = True)
        >>> theta, phi = sympy.Symbol("theta"), sympy.Symbol("phi")
        >>> assert gf.convert(cirq.FSimGate(-np.pi/4, phi), cirq.ISwapPowGate) == cirq.SQRT_ISWAP
        >>> assert gf.convert(cirq.FSimGate(theta, np.pi/4), cirq.ISwapPowGate) is None

        Args:
            gate            : `cirq.Gate` instance to convert.
            target_gate_type: One of `POSSIBLE_FSIM_GATES` types to which the given gate should be
                              converted to.
        Returns:
            The converted gate instances if the converion is possible, else None.
        Raises:
            ValueError: If `target_gate_type` is not one of `POSSIBLE_FSIM_GATES`.
        """
        if target_gate_type not in self._supported_types:
            raise ValueError(f"{target_gate_type} must be one of {self._supported_types}")
        if not self.allow_symbols and cirq.is_parameterized(gate):
            return None
        return cast(T, self._supported_types[target_gate_type](cast(POSSIBLE_FSIM_GATES, gate)))

    def _convert_to_fsim(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.FSimGate]:
        theta = phi = None
        if isinstance(g, cirq.FSimGate) or (
            isinstance(g, cirq.PhasedFSimGate)
            and self._approx_eq_or_symbol(g._value_equality_values_()[1:4], (0.0, 0.0, 0.0))
        ):
            theta = g.theta
            phi = g.phi
        if isinstance(g, cirq.PhasedISwapPowGate) and self._approx_eq_or_symbol(
            g.phase_exponent, 0
        ):
            g = g._iswap
        if isinstance(g, (cirq.ISwapPowGate, cirq.CZPowGate)):
            if not self._approx_eq_or_symbol(_exp(np.pi * 1j * g.global_shift * g.exponent), 1.0):
                return None
            theta = -g.exponent * np.pi / 2 if isinstance(g, cirq.ISwapPowGate) else 0
            phi = -g.exponent * np.pi if isinstance(g, cirq.CZPowGate) else 0
        if isinstance(g, cirq.IdentityGate):
            theta = phi = 0
        return None if (theta is None or phi is None) else cirq.FSimGate(theta, phi)

    def _convert_to_phased_fsim(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.PhasedFSimGate]:
        if isinstance(g, cirq.PhasedFSimGate):
            return g
        chi = 0
        if isinstance(g, cirq.PhasedISwapPowGate):
            chi = g.phase_exponent * 2 * np.pi
            g = g._iswap
        fsim = self._convert_to_fsim(g)
        return None if fsim is None else cirq.PhasedFSimGate(fsim.theta, 0, chi, 0, fsim.phi)

    def _convert_to_iswap(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.ISwapPowGate]:
        if isinstance(g, cirq.ISwapPowGate):
            return g
        if isinstance(g, cirq.PhasedISwapPowGate):
            return g._iswap if self._approx_eq_or_symbol(g.phase_exponent, 0) else None
        fsim = self._convert_to_fsim(g)
        return (
            None
            if (fsim is None or not self._approx_eq_or_symbol(fsim.phi, 0))
            else cirq.ISWAP ** (-2 * fsim.theta / np.pi)
        )

    def _convert_to_phased_iswap(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.PhasedISwapPowGate]:
        if isinstance(g, cirq.PhasedISwapPowGate):
            return g
        if isinstance(g, cirq.PhasedFSimGate) and self._approx_eq_or_symbol(
            (g.zeta, g.gamma, g.phi), (0, 0, 0)
        ):
            return cirq.PhasedISwapPowGate(
                exponent=-2 * g.theta / np.pi, phase_exponent=g.chi / (2 * np.pi)
            )
        cg = self._convert_to_iswap(g)
        return (
            None if cg is None else cirq.PhasedISwapPowGate(exponent=cg.exponent, phase_exponent=0)
        )

    def _convert_to_cz(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.CZPowGate]:
        if isinstance(g, cirq.CZPowGate):
            return g
        cg = self._convert_to_fsim(g)
        return (
            None
            if (cg is None or not self._approx_eq_or_symbol(cg.theta, 0))
            else cirq.CZ ** (-cg.phi / np.pi)
        )

    def _convert_to_identity(self, g: POSSIBLE_FSIM_GATES) -> Optional[cirq.IdentityGate]:
        cg = self._convert_to_fsim(g)
        return (
            None
            if (cg is None or not self._approx_eq_or_symbol((cg.theta, cg.phi), (0, 0)))
            else cirq.IdentityGate(2)
        )
