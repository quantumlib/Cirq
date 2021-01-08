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
import re
from fractions import Fraction
from typing import (
    Any,
    TYPE_CHECKING,
    Optional,
    Union,
    TypeVar,
    Dict,
    overload,
    Iterable,
    List,
    Sequence,
)

import numpy as np
import sympy
from typing_extensions import Protocol

from cirq import protocols, value
from cirq._doc import doc_private

if TYPE_CHECKING:
    import cirq


@value.value_equality
class CircuitDiagramInfo:
    """Describes how to draw an operation in a circuit diagram."""

    def __init__(
        self,
        wire_symbols: Iterable[str],
        exponent: Any = 1,
        connected: bool = True,
        exponent_qubit_index: Optional[int] = None,
        auto_exponent_parens: bool = True,
    ) -> None:
        """
        Args:
            wire_symbols: The symbols that should be shown on the qubits
                affected by this operation. Must match the number of qubits that
                the operation is applied to.
            exponent: An optional convenience value that will be appended onto
                an operation's final gate symbol with a caret in front
                (unless it's equal to 1). For example, the square root of X gate
                has a text diagram exponent of 0.5 and symbol of 'X' so it is
                drawn as 'X^0.5'.
            connected: Whether or not to draw a line connecting the qubits.
            exponent_qubit_index: The qubit to put the exponent on. (The k'th
                qubit is the k'th target of the gate.) Defaults to the bottom
                qubit in the diagram.
            auto_exponent_parens: When this is True, diagram making code will
                add parentheses around exponents whose contents could look
                ambiguous (e.g. if the exponent contains a dash character that
                could be mistaken for an identity wire). Defaults to True.
        """
        if isinstance(wire_symbols, str):
            raise ValueError('Expected an Iterable[str] for wire_symbols but got a str.')
        self.wire_symbols = tuple(wire_symbols)
        self.exponent = exponent
        self.connected = connected
        self.exponent_qubit_index = exponent_qubit_index
        self.auto_exponent_parens = auto_exponent_parens

    def with_wire_symbols(self, new_wire_symbols: Iterable[str]):
        return CircuitDiagramInfo(
            wire_symbols=new_wire_symbols,
            exponent=self.exponent,
            connected=self.connected,
            exponent_qubit_index=self.exponent_qubit_index,
            auto_exponent_parens=self.auto_exponent_parens,
        )

    def _value_equality_values_(self) -> Any:
        return (
            self.wire_symbols,
            self.exponent,
            self.connected,
            self.exponent_qubit_index,
            self.auto_exponent_parens,
        )

    def _wire_symbols_including_formatted_exponent(
        self, args: 'cirq.CircuitDiagramInfoArgs', *, preferred_exponent_index: Optional[int] = None
    ) -> List[str]:
        result = list(self.wire_symbols)
        exponent = self._formatted_exponent(args)
        if exponent is not None:
            ks: Sequence[int]
            if self.exponent_qubit_index is not None:
                ks = (self.exponent_qubit_index,)
            elif not self.connected:
                ks = range(len(result))
            elif preferred_exponent_index is not None:
                ks = (preferred_exponent_index,)
            else:
                ks = (0,)
            for k in ks:
                result[k] += '^' + exponent
        return result

    def _formatted_exponent(
        self: 'cirq.CircuitDiagramInfo', args: 'cirq.CircuitDiagramInfoArgs'
    ) -> Optional[str]:

        if protocols.is_parameterized(self.exponent):
            name = str(self.exponent)
            return '({})'.format(name) if _is_exposed_formula(name) else name

        if self.exponent == 0:
            return '0'

        # 1 is not shown.
        if self.exponent == 1:
            return None

        # Round -1.0 into -1.
        if self.exponent == -1:
            return '-1'

        # If it's a float, show the desired precision.
        if isinstance(self.exponent, float):
            if args.precision is not None:
                # funky behavior of fraction, cast to str in constructor helps.
                approx_frac = Fraction(self.exponent).limit_denominator(16)
                if approx_frac.denominator not in [2, 4, 5, 10]:
                    if abs(float(approx_frac) - self.exponent) < 10 ** -args.precision:
                        return '({})'.format(approx_frac)

                return args.format_real(self.exponent)
            return repr(self.exponent)

        # If the exponent is any other object, use its string representation.
        s = str(self.exponent)
        if self.auto_exponent_parens and ('+' in s or ' ' in s or '-' in s[1:]):
            # The string has confusing characters. Put parens around it.
            return '({})'.format(self.exponent)
        return s

    @staticmethod
    def _op_info_with_fallback(
        op: 'cirq.Operation', args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        info = protocols.circuit_diagram_info(op, args, None)
        if info is not None:
            if len(op.qubits) != len(info.wire_symbols):
                raise ValueError(
                    'Wanted diagram info from {!r} for {} '
                    'qubits but got {!r}'.format(op, len(op.qubits), info)
                )
            return info

        # Use the untagged operation's __str__.
        name = str(op.untagged)

        # Representation usually looks like 'gate(qubit1, qubit2, etc)'.
        # Try to cut off the qubit part, since that would be redundant.
        redundant_tail = '({})'.format(', '.join(str(e) for e in op.qubits))
        if name.endswith(redundant_tail):
            name = name[: -len(redundant_tail)]

        # Add tags onto the representation, if they exist
        if op.tags:
            name += f'{list(op.tags)}'

        # Include ordering in the qubit labels.
        symbols = (name,) + tuple('#{}'.format(i + 1) for i in range(1, len(op.qubits)))

        return protocols.CircuitDiagramInfo(wire_symbols=symbols)

    def __repr__(self) -> str:
        return (
            'cirq.CircuitDiagramInfo('
            f'wire_symbols={self.wire_symbols!r}, '
            f'exponent={self.exponent!r}, '
            f'connected={self.connected!r}, '
            f'exponent_qubit_index={self.exponent_qubit_index!r}, '
            f'auto_exponent_parens={self.auto_exponent_parens!r})'
        )


def _is_exposed_formula(text: str) -> bool:
    return re.match('[a-zA-Z_][a-zA-Z0-9_]*$', text) is None


@value.value_equality
class CircuitDiagramInfoArgs:
    """A request for information on drawing an operation in a circuit diagram.

    Attributes:
        known_qubits: The qubits the gate is being applied to. None means this
            information is not known by the caller.
        known_qubit_count: The number of qubits the gate is being applied to
            None means this information is not known by the caller.
        use_unicode_characters: If true, the wire symbols are permitted to
            include unicode characters (as long as they work well in fixed
            width fonts). If false, use only ascii characters. ASCII is
            preferred in cases where UTF8 support is done poorly, or where
            the fixed-width font being used to show the diagrams does not
            properly handle unicode characters.
        precision: The number of digits after the decimal to show for numbers in
            the text diagram. None means use full precision.
        qubit_map: The map from qubits to diagram positions.
        include_tags: Whether to print tags from TaggedOperations
    """

    UNINFORMED_DEFAULT = None  # type: CircuitDiagramInfoArgs

    def __init__(
        self,
        known_qubits: Optional[Iterable['cirq.Qid']],
        known_qubit_count: Optional[int],
        use_unicode_characters: bool,
        precision: Optional[int],
        qubit_map: Optional[Dict['cirq.Qid', int]],
        include_tags: bool = True,
    ) -> None:
        self.known_qubits = None if known_qubits is None else tuple(known_qubits)
        self.known_qubit_count = known_qubit_count
        self.use_unicode_characters = use_unicode_characters
        self.precision = precision
        self.qubit_map = qubit_map
        self.include_tags = include_tags

    def _value_equality_values_(self) -> Any:
        return (
            self.known_qubits,
            self.known_qubit_count,
            self.use_unicode_characters,
            self.precision,
            None
            if self.qubit_map is None
            else tuple(sorted(self.qubit_map.items(), key=lambda e: e[0])),
            self.include_tags,
        )

    def __repr__(self) -> str:
        return (
            'cirq.CircuitDiagramInfoArgs('
            f'known_qubits={self.known_qubits!r}, '
            f'known_qubit_count={self.known_qubit_count!r}, '
            f'use_unicode_characters={self.use_unicode_characters!r}, '
            f'precision={self.precision!r}, '
            f'qubit_map={self.qubit_map!r},'
            f'include_tags={self.include_tags!r})'
        )

    def format_real(self, val: Union[sympy.Basic, int, float]) -> str:
        if isinstance(val, sympy.Basic):
            return str(val)
        if val == int(val):
            return str(int(val))
        if self.precision is None:
            return str(val)
        return f'{float(val):.{self.precision}}'

    def format_complex(self, val: Union[sympy.Basic, int, float, complex]) -> str:
        if isinstance(val, sympy.Basic):
            return str(val)
        c = complex(val)
        joiner = '+'
        abs_imag = c.imag
        if abs_imag < 0:
            joiner = '-'
            abs_imag *= -1
        imag_str = '' if abs_imag == 1 else self.format_real(abs_imag)
        return f'{self.format_real(c.real)}{joiner}{imag_str}i'

    def format_radians(self, radians: Union[sympy.Basic, int, float]) -> str:
        """Returns angle in radians as a human-readable string."""
        if protocols.is_parameterized(radians):
            return str(radians)
        unit = 'π' if self.use_unicode_characters else 'pi'
        if radians == np.pi:
            return unit
        if radians == 0:
            return '0'
        if radians == -np.pi:
            return '-' + unit
        if self.precision is not None:
            quantity = self.format_real(radians / np.pi)
            return quantity + unit
        return repr(radians)

    def copy(self):
        return self.__class__(
            known_qubits=self.known_qubits,
            known_qubit_count=self.known_qubit_count,
            use_unicode_characters=self.use_unicode_characters,
            precision=self.precision,
            qubit_map=self.qubit_map,
        )

    def with_args(self, **kwargs):
        args = self.copy()
        for arg_name, val in kwargs.items():
            setattr(args, arg_name, val)
        return args


CircuitDiagramInfoArgs.UNINFORMED_DEFAULT = CircuitDiagramInfoArgs(
    known_qubits=None,
    known_qubit_count=None,
    use_unicode_characters=True,
    precision=3,
    qubit_map=None,
)


class SupportsCircuitDiagramInfo(Protocol):
    """A diagrammable operation on qubits."""

    @doc_private
    def _circuit_diagram_info_(
        self, args: CircuitDiagramInfoArgs
    ) -> Union[str, Iterable[str], CircuitDiagramInfo]:
        """Describes how to draw an operation in a circuit diagram.

        This method is used by the global `cirq.diagram_info` method. If this
        method is not present, or returns NotImplemented, it is assumed that the
        receiving object doesn't specify diagram info.

        Args:
            args: A DiagramInfoArgs instance encapsulating various pieces of
                information (e.g. how many qubits are we being applied to) as
                well as user options (e.g. whether to avoid unicode characters).

        Returns:
            A DiagramInfo instance describing what to show.
        """


TDefault = TypeVar('TDefault')
RaiseTypeErrorIfNotProvided = CircuitDiagramInfo(())


# pylint: disable=function-redefined
@overload
def circuit_diagram_info(
    val: Any,
    args: Optional[CircuitDiagramInfoArgs] = None,
) -> CircuitDiagramInfo:
    pass


@overload
def circuit_diagram_info(
    val: Any, args: Optional[CircuitDiagramInfoArgs], default: TDefault
) -> Union[CircuitDiagramInfo, TDefault]:
    pass


@overload
def circuit_diagram_info(val: Any, *, default: TDefault) -> Union[CircuitDiagramInfo, TDefault]:
    pass


def circuit_diagram_info(
    val: Any, args: Optional[CircuitDiagramInfoArgs] = None, default=RaiseTypeErrorIfNotProvided
):
    """Requests information on drawing an operation in a circuit diagram.

    Calls _circuit_diagram_info_ on `val`. If `val` doesn't have
    _circuit_diagram_info_, or it returns NotImplemented, that indicates that
    diagram information is not available.

    Args:
        val: The operation or gate that will need to be drawn.
        args: A CircuitDiagramInfoArgs describing the desired drawing style.
        default: A default result to return if the value doesn't have circuit
            diagram information. If not specified, a TypeError is raised
            instead.

    Returns:
        If `val` has no _circuit_diagram_info_ method or it returns
        NotImplemented, then `default` is returned (or a TypeError is
        raised if no `default` is specified).

        Otherwise, the value returned by _circuit_diagram_info_ is returned.

    Raises:
        TypeError:
            `val` doesn't have circuit diagram information and `default` was
            not specified.
    """

    # Attempt.
    if args is None:
        args = CircuitDiagramInfoArgs.UNINFORMED_DEFAULT
    getter = getattr(val, '_circuit_diagram_info_', None)
    result = NotImplemented if getter is None else getter(args)

    # Success?
    if isinstance(result, str):
        return CircuitDiagramInfo(wire_symbols=(result,))
    if isinstance(result, Iterable):
        return CircuitDiagramInfo(wire_symbols=tuple(result))
    if result is not NotImplemented:
        return result

    # Failure.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if getter is None:
        raise TypeError(
            "object of type '{}' has no _circuit_diagram_info_ method.".format(type(val))
        )
    raise TypeError(
        "object of type '{}' does have a _circuit_diagram_info_ "
        "method, but it returned NotImplemented.".format(type(val))
    )


# pylint: enable=function-redefined
