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

from typing import Callable, Dict, Iterable, List, overload, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from cirq import protocols
from cirq.ops import raw_types, pauli_string
from cirq.ops.measurement_gate import MeasurementGate
from cirq.ops.pauli_measurement_gate import PauliMeasurementGate

if TYPE_CHECKING:
    import cirq


def _default_measurement_key(qubits: Iterable[raw_types.Qid]) -> str:
    return ','.join(str(q) for q in qubits)


def measure_single_paulistring(
    pauli_observable: pauli_string.PauliString,
    key: Optional[Union[str, 'cirq.MeasurementKey']] = None,
) -> raw_types.Operation:
    """Returns a single PauliMeasurementGate which measures the pauli observable

    Args:
        pauli_observable: The `cirq.PauliString` observable to measure.
        key: Optional `str` or `cirq.MeasurementKey` that gate should use.
            If none provided, it defaults to a comma-separated list of
            `str(qubit)` for each of the target qubits.

    Returns:
        An operation measuring the pauli observable.

    Raises:
        ValueError: if the observable is not an instance of PauliString or if the coefficient
            is not +1 or -1.
    """
    if not isinstance(pauli_observable, pauli_string.PauliString):
        raise ValueError(
            f'Pauli observable {pauli_observable} should be an instance of cirq.PauliString.'
        )
    if abs(pauli_observable.coefficient) != 1:
        raise ValueError(
            f"Pauli observable {pauli_observable} must have a coefficient of +1 or -1."
        )

    if key is None:
        key = _default_measurement_key(pauli_observable)
    return PauliMeasurementGate(pauli_observable.dense(list(pauli_observable.keys())), key).on(
        *pauli_observable.keys()
    )


def measure_paulistring_terms(
    pauli_basis: pauli_string.PauliString, key_func: Callable[[raw_types.Qid], str] = str
) -> List[raw_types.Operation]:
    """Returns a list of operations individually measuring qubits in the pauli basis.

    Args:
        pauli_basis: The `cirq.PauliString` basis in which each qubit should
            be measured.
        key_func: Determines the key of the measurements of each qubit. Takes
            the qubit and returns the key for that qubit. Defaults to str.

    Returns:
        A list of operations individually measuring the given qubits in the
        specified pauli basis.

    Raises:
        ValueError: if `pauli_basis` is not an instance of `cirq.PauliString`.
    """
    if not isinstance(pauli_basis, pauli_string.PauliString):
        raise ValueError(
            f'Pauli observable {pauli_basis} should be an instance of cirq.PauliString.'
        )
    return [PauliMeasurementGate([pauli_basis[q]], key=key_func(q)).on(q) for q in pauli_basis]


# pylint: disable=function-redefined


@overload
def measure(
    *target: raw_types.Qid,
    key: Optional[Union[str, 'cirq.MeasurementKey']] = None,
    invert_mask: Tuple[bool, ...] = (),
) -> raw_types.Operation:
    pass


@overload
def measure(
    __target: Iterable[raw_types.Qid],
    *,
    key: Optional[Union[str, 'cirq.MeasurementKey']] = None,
    invert_mask: Tuple[bool, ...] = (),
) -> raw_types.Operation:
    pass


def measure(
    *target,
    key: Optional[Union[str, 'cirq.MeasurementKey']] = None,
    invert_mask: Tuple[bool, ...] = (),
    confusion_map: Optional[Dict[Tuple[int, ...], np.ndarray]] = None,
) -> raw_types.Operation:
    """Returns a single MeasurementGate applied to all the given qubits.

    The qubits are measured in the computational basis. This can also be
    used with the alias `cirq.M`.

    Args:
        *target: The qubits that the measurement gate should measure.
            These can be specified as separate function arguments or
            with a single argument for an iterable of qubits.
        key: Optional `str` or `cirq.MeasurementKey` that gate should use.
            If none provided, it defaults to a comma-separated list of
            `str(qubit)` for each of the target qubits.
        invert_mask: A list of Truthy or Falsey values indicating whether
            the corresponding qubits should be flipped. None indicates no
            inverting should be done.
        confusion_map: A map of qubit index sets (using indices in
            `target`) to the 2D confusion matrix for those qubits. Indices
            not included use the identity. Applied before invert_mask if both
            are provided.

    Returns:
        An operation targeting the given qubits with a measurement.

    Raises:
        ValueError: If the qubits are not instances of Qid.
    """
    one_iterable_arg: bool = (
        len(target) == 1
        and isinstance(target[0], Iterable)
        and not isinstance(target[0], (bytes, str, np.ndarray))
    )
    targets = tuple(target[0]) if one_iterable_arg else target
    for qubit in targets:
        if isinstance(qubit, np.ndarray):
            raise ValueError(
                'measure() was called a numpy ndarray. Perhaps you meant '
                'to call measure_state_vector on numpy array?'
            )
        elif not isinstance(qubit, raw_types.Qid):
            raise ValueError('measure() was called with type different than Qid.')

    if key is None:
        key = _default_measurement_key(targets)
    qid_shape = protocols.qid_shape(targets)
    return MeasurementGate(len(targets), key, invert_mask, qid_shape, confusion_map).on(*targets)


M = measure


@overload
def measure_each(
    *qubits: raw_types.Qid, key_func: Callable[[raw_types.Qid], str] = str
) -> List[raw_types.Operation]:
    pass


@overload
def measure_each(
    __qubits: Iterable[raw_types.Qid], *, key_func: Callable[[raw_types.Qid], str] = str
) -> List[raw_types.Operation]:
    pass


def measure_each(
    *qubits, key_func: Callable[[raw_types.Qid], str] = str
) -> List[raw_types.Operation]:
    """Returns a list of operations individually measuring the given qubits.

    The qubits are measured in the computational basis.

    Args:
        *qubits: The qubits to measure.  These can be passed as separate
            function arguments or as a one-argument iterable of qubits.
        key_func: Determines the key of the measurements of each qubit. Takes
            the qubit and returns the key for that qubit. Defaults to str.

    Returns:
        A list of operations individually measuring the given qubits.
    """
    one_iterable_arg: bool = (
        len(qubits) == 1
        and isinstance(qubits[0], Iterable)
        and not isinstance(qubits[0], (bytes, str))
    )
    qubitsequence = qubits[0] if one_iterable_arg else qubits
    return [MeasurementGate(1, key_func(q), qid_shape=(q.dimension,)).on(q) for q in qubitsequence]


# pylint: enable=function-redefined
