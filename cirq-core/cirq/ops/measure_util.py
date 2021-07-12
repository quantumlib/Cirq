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

from typing import Callable, Iterable, List, Optional, Tuple, TYPE_CHECKING, Union, Any

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types
from cirq.ops.measurement_gate import MeasurementGate

if TYPE_CHECKING:
    import cirq


def _default_measurement_key(qubits: Iterable[raw_types.Qid]) -> str:
    return ','.join(str(q) for q in qubits)


def _get_measurement(
    target: Union['cirq.Qid', Iterable['cirq.Qid']],
    key: Optional[Union[str, value.MeasurementKey]] = None,
    invert_mask: Tuple[bool, ...] = (),
) -> Union[raw_types.Operation, List[raw_types.Operation]]:
    if key is None:
        key = _default_measurement_key(target)
    qid_shape = protocols.qid_shape(target)
    return MeasurementGate(len(target), key, invert_mask, qid_shape).on(*target)


def _get_each_measurement(
    target: Iterable['cirq.Qid'], key_func: Callable[[raw_types.Qid], str]
) -> List[raw_types.Operation]:
    return [MeasurementGate(1, key_func(q), qid_shape=(q.dimension,)).on(q) for q in target]


def measure(
    *target: Union['cirq.Qid', Iterable['cirq.Qid']],
    key: Optional[Union[str, value.MeasurementKey]] = None,
    invert_mask: Tuple[bool, ...] = (),
) -> Union[raw_types.Operation, List[raw_types.Operation]]:
    """Returns a single MeasurementGate applied to all the given qubits
     or a list given a list of qubits.

    The qubits are measured in the computational basis.

    Args:
        *target: The qubits that the measurement gate should measure.
        key: The string key of the measurement. If this is None, it defaults
            to a comma-separated list of the target qubits' str values.
        invert_mask: A list of Truthy or Falsey values indicating whether
            the corresponding qubits should be flipped. None indicates no
            inverting should be done.

    Returns:
        An operation targeting the given qubits with a measurement.

    Raises:
        ValueError if the qubits are not instances of Qid or list of Qid.
    """
    for qubit in target:
        if isinstance(qubit, list):
            return _get_measurement(qubit, key=key, invert_mask=invert_mask)
        if isinstance(qubit, np.ndarray):
            raise ValueError(
                'measure() was called a numpy ndarray. Perhaps you meant '
                'to call measure_state_vector on numpy array?'
            )
        elif not isinstance(qubit, raw_types.Qid):
            raise ValueError('measure() was called with type different than Qid.')

    return _get_measurement(target, key=key, invert_mask=invert_mask)


def measure_each(
    *qubits: Union['cirq.Qid', Iterable['cirq.Qid']], key_func: Callable[[raw_types.Qid], str] = str
) -> List[raw_types.Operation]:
    """Returns a list of operations individually measuring the given qubits.

    The qubits are measured in the computational basis.

    Args:
        *qubits: The qubits or list of qubits to measure.
        key_func: Determines the key of the measurements of each qubit. Takes
            the qubit and returns the key for that qubit. Defaults to str.

    Returns:
        A list of operations individually measuring the given qubits.
    """
    if len(qubits) == 1 and isinstance(qubits[0], list):
        return _get_each_measurement(qubits[0], key_func)

    return _get_each_measurement(qubits, key_func)
