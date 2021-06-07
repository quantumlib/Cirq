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

import string
from typing import TYPE_CHECKING, Union, Any, Tuple, TypeVar, Optional, Dict, Iterable

from typing_extensions import Protocol

from cirq import ops
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

TDefault = TypeVar('TDefault')

RaiseTypeErrorIfNotProvided = ([],)  # type: Any


class QasmArgs(string.Formatter):
    def __init__(
        self,
        precision: int = 10,
        version: str = '2.0',
        qubit_id_map: Dict['cirq.Qid', str] = None,
        meas_key_id_map: Dict[str, str] = None,
    ) -> None:
        """
        Args:
            precision: The number of digits after the decimal to show for
                numbers in the qasm code.
            version: The QASM version to target. Objects may return different
                qasm depending on version.
            qubit_id_map: A dictionary mapping qubits to qreg QASM identifiers.
            meas_key_id_map: A dictionary mapping measurement keys to creg QASM
                identifiers.
        """
        self.precision = precision
        self.version = version
        self.qubit_id_map = {} if qubit_id_map is None else qubit_id_map
        self.meas_key_id_map = {} if meas_key_id_map is None else meas_key_id_map

    def format_field(self, value: Any, spec: str) -> str:
        """Method of string.Formatter that specifies the output of format()."""
        if isinstance(value, (float, int)):
            if isinstance(value, float):
                value = round(value, self.precision)
            if spec == 'half_turns':
                value = f'pi*{value}' if value != 0 else '0'
                spec = ''
        elif isinstance(value, ops.Qid):
            value = self.qubit_id_map[value]
        elif isinstance(value, str) and spec == 'meas':
            value = self.meas_key_id_map[value]
            spec = ''
        return super().format_field(value, spec)

    def validate_version(self, *supported_versions: str) -> None:
        if self.version not in supported_versions:
            raise ValueError(f'QASM version {self.version} output is not supported.')


class SupportsQasm(Protocol):
    """An object that can be turned into QASM code.

    Returning `NotImplemented` or `None` means "don't know how to turn into
    QASM". In that case fallbacks based on decomposition and known unitaries
    will be used instead.
    """

    @doc_private
    def _qasm_(self) -> Union[None, NotImplementedType, str]:
        pass


class SupportsQasmWithArgs(Protocol):
    """An object that can be turned into QASM code.

    Returning `NotImplemented` or `None` means "don't know how to turn into
    QASM". In that case fallbacks based on decomposition and known unitaries
    will be used instead.
    """

    @doc_private
    def _qasm_(self, args: QasmArgs) -> Union[None, NotImplementedType, str]:
        pass


class SupportsQasmWithArgsAndQubits(Protocol):
    """An object that can be turned into QASM code if it knows its qubits.

    Returning `NotImplemented` or `None` means "don't know how to turn into
    QASM". In that case fallbacks based on decomposition and known unitaries
    will be used instead.
    """

    @doc_private
    def _qasm_(
        self, qubits: Tuple['cirq.Qid'], args: QasmArgs
    ) -> Union[None, NotImplementedType, str]:
        pass


# pylint: disable=function-redefined
def qasm(
    val: Any,
    *,
    args: Optional[QasmArgs] = None,
    qubits: Optional[Iterable['cirq.Qid']] = None,
    default: TDefault = RaiseTypeErrorIfNotProvided,
) -> Union[str, TDefault]:
    """Returns QASM code for the given value, if possible.

    Different values require different sets of arguments. The general rule of
    thumb is that circuits don't need any, operations need a `QasmArgs`, and
    gates need both a `QasmArgs` and `qubits`.

    Args:
        val: The value to turn into QASM code.
        args: A `QasmArgs` object to pass into the value's `_qasm_` method.
            This is for needed for objects that only have a local idea of what's
            going on, e.g. a `cirq.Operation` in a bigger `cirq.Circuit`
            involving qubits that the operation wouldn't otherwise know about.
        qubits: A list of qubits that the value is being applied to. This is
            needed for `cirq.Gate` values, which otherwise wouldn't know what
            qubits to talk about.
        default: A default result to use if the value doesn't have a
            `_qasm_` method or that method returns `NotImplemented` or `None`.
            If not specified, non-decomposable values cause a `TypeError`.

    Returns:
        The result of `val._qasm_(...)`, if `val` has a `_qasm_`
        method and it didn't return `NotImplemented` or `None`. Otherwise
        `default` is returned, if it was specified. Otherwise an error is
        raised.

    TypeError:
        `val` didn't have a `_qasm_` method (or that method returned
        `NotImplemented` or `None`) and `default` wasn't set.
    """
    method = getattr(val, '_qasm_', None)
    result = NotImplemented
    if method is not None:
        kwargs = {}  # type: Dict[str, Any]
        if args is not None:
            kwargs['args'] = args
        if qubits is not None:
            kwargs['qubits'] = tuple(qubits)
        result = method(**kwargs)
    if result is not None and result is not NotImplemented:
        return result

    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if method is None:
        raise TypeError(f"object of type '{type(val)}' has no _qasm_ method.")
    raise TypeError(
        "object of type '{}' does have a _qasm_ method, "
        "but it returned NotImplemented or None.".format(type(val))
    )


# pylint: enable=function-redefined
