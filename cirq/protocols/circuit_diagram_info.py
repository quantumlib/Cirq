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
import collections
from typing import Any, TYPE_CHECKING, Optional, Union, Tuple, TypeVar, Dict, \
    overload, Iterable

from typing_extensions import Protocol

from cirq import value

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


@value.value_equality
class CircuitDiagramInfo:
    """Describes how to draw an operation in a circuit diagram."""

    def __init__(self,
                 wire_symbols: Tuple[str, ...],
                 exponent: Any = 1,
                 connected: bool = True) -> None:
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
        """
        if isinstance(wire_symbols, str):
            raise ValueError(
                'Expected a Tuple[str] for wire_symbols but got a str.')
        self.wire_symbols = wire_symbols
        self.exponent = exponent
        self.connected = connected

    def _value_equality_values_(self):
        return self.wire_symbols, self.exponent, self.connected

    def __repr__(self):
        return ('cirq.CircuitDiagramInfo(' +
                'wire_symbols={!r}, '.format(self.wire_symbols) +
                'exponent={!r}, '.format(self.exponent) +
                'connected={!r})'.format(self.connected)
                )


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
    """

    UNINFORMED_DEFAULT = None  # type: CircuitDiagramInfoArgs

    def __init__(self,
                 known_qubits: Optional[Iterable['cirq.QubitId']],
                 known_qubit_count: Optional[int],
                 use_unicode_characters: bool,
                 precision: Optional[int],
                 qubit_map: Optional[Dict['cirq.QubitId', int]]) -> None:
        self.known_qubits = (None if known_qubits is None
                             else tuple(known_qubits))
        self.known_qubit_count = known_qubit_count
        self.use_unicode_characters = use_unicode_characters
        self.precision = precision
        self.qubit_map = qubit_map

    def _value_equality_values_(self):
        return (self.known_qubits,
                self.known_qubit_count,
                self.use_unicode_characters,
                self.precision,
                None
                if self.qubit_map is None else
                tuple(sorted(self.qubit_map.items(), key=lambda e: e[0])))

    def __repr__(self):
        return (
            'cirq.CircuitDiagramInfoArgs('
            'known_qubits={!r}, '
            'known_qubit_count={!r}, '
            'use_unicode_characters={!r}, '
            'precision={!r}, '
            'qubit_map={!r})'.format(
                self.known_qubits,
                self.known_qubit_count,
                self.use_unicode_characters,
                self.precision,
                self.qubit_map))


CircuitDiagramInfoArgs.UNINFORMED_DEFAULT = CircuitDiagramInfoArgs(
    known_qubits=None,
    known_qubit_count=None,
    use_unicode_characters=True,
    precision=3,
    qubit_map=None)


class SupportsCircuitDiagramInfo(Protocol):
    """A diagrammable operation on qubits."""

    def _circuit_diagram_info_(self, args: CircuitDiagramInfoArgs
                               ) -> Union[str,
                                          Iterable[str],
                                          CircuitDiagramInfo]:
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
def circuit_diagram_info(val: Any,
                         args: Optional[CircuitDiagramInfoArgs] = None,
                         ) -> CircuitDiagramInfo:
    pass


@overload
def circuit_diagram_info(val: Any,
                         args: Optional[CircuitDiagramInfoArgs],
                         default: TDefault
                         ) -> Union[CircuitDiagramInfo, TDefault]:
    pass


@overload
def circuit_diagram_info(val: Any,
                         *,
                         default: TDefault
                         ) -> Union[CircuitDiagramInfo, TDefault]:
    pass


def circuit_diagram_info(val: Any,
                         args: Optional[CircuitDiagramInfoArgs] = None,
                         default=RaiseTypeErrorIfNotProvided):
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
    if isinstance(result, collections.Iterable):
        return CircuitDiagramInfo(wire_symbols=tuple(result))
    if result is not NotImplemented:
        return result

    # Failure.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if getter is None:
        raise TypeError(
            "object of type '{}' "
            "has no _circuit_diagram_info_ method.".format(type(val)))
    raise TypeError("object of type '{}' does have a _circuit_diagram_info_ "
                    "method, but it returned NotImplemented.".format(type(val)))
# pylint: enable=function-redefined
