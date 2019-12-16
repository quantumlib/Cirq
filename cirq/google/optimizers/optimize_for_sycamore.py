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
"""A combination of several optimizations targeting XmonDevice."""
from typing import Callable, cast, List, Optional, TYPE_CHECKING

from cirq import circuits, devices, optimizers
from cirq.google.optimizers import (convert_to_xmon_gates,
                                    ConvertToSycamoreGates,
                                    ConvertToSqrtIswapGates)

if TYPE_CHECKING:
    import cirq

_TOLERANCE = 1e-5


def _merge_rots(c: 'cirq.Circuit'):
    return optimizers.merge_single_qubit_gates_into_phased_x_z(c, _TOLERANCE)


_XMON_OPTIMIZERS: List[Callable[['cirq.Circuit'], None]] = [
    convert_to_xmon_gates.ConvertToXmonGates().optimize_circuit,
    optimizers.MergeInteractions(tolerance=_TOLERANCE,
                                 allow_partial_czs=False).optimize_circuit,
    _merge_rots,
    optimizers.EjectPhasedPaulis(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.EjectZ(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.DropNegligible(tolerance=_TOLERANCE).optimize_circuit,
]

_XMON_OPTIMIZERS_PART_CZ: List[Callable[['cirq.Circuit'], None]] = [
    convert_to_xmon_gates.ConvertToXmonGates().optimize_circuit,
    optimizers.MergeInteractions(tolerance=_TOLERANCE,
                                 allow_partial_czs=True).optimize_circuit,
    _merge_rots,
    optimizers.EjectPhasedPaulis(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.EjectZ(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.DropNegligible(tolerance=_TOLERANCE).optimize_circuit,
]

_SYCAMORE_OPTIMIZERS = [
    ConvertToSycamoreGates().optimize_circuit,
    _merge_rots,
    optimizers.EjectPhasedPaulis(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.EjectZ(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.DropNegligible(tolerance=_TOLERANCE).optimize_circuit,
]  # type: List[Callable[[circuits.Circuit], None]]

_SQRT_ISWAP_OPTIMIZERS = [
    ConvertToSqrtIswapGates().optimize_circuit,
    _merge_rots,
    optimizers.EjectPhasedPaulis(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.EjectZ(tolerance=_TOLERANCE).optimize_circuit,
    optimizers.DropNegligible(tolerance=_TOLERANCE).optimize_circuit,
]  # type: List[Callable[[circuits.Circuit], None]]

_OPTIMIZER_TYPES = {
    'xmon': _XMON_OPTIMIZERS,
    'xmon_partial_cz': _XMON_OPTIMIZERS_PART_CZ,
    'sqrt_iswap': _SQRT_ISWAP_OPTIMIZERS,
    'sycamore': _SYCAMORE_OPTIMIZERS,
}


def optimized_for_sycamore(
        circuit: 'cirq.Circuit',
        new_device: Optional['cirq.google.XmonDevice'] = None,
        qubit_map: Callable[['cirq.Qid'], devices.GridQubit] = lambda e: cast(
            devices.GridQubit, e),
        optimizer_type: str = 'sqrt_iswap') -> 'cirq.Circuit':
    """Optimizes a circuit for Google devices.

    Uses a set of optimizers that will compile to the proper gateset for the
    device (xmon, sqrt_iswap, or sycamore gates) and then use optimizers to
    compresss the gate depth down as much as is easily algorithmically possible
    by merging rotations, ejecting Z gates, etc.

    Args:
        circuit: The circuit to optimize.
        new_device: The device the optimized circuit should be targeted at. If
            set to None, the circuit's current device is used.
        qubit_map: Transforms the qubits (e.g. so that they are GridQubits).
        optimizer_type: A string defining the optimizations to apply.
            Possible values are  'xmon', 'xmon_partial_cz', 'sqrt_iswap',
            'sycamore'
    Returns:
        The optimized circuit.
    """
    copy = circuit.copy()
    if optimizer_type not in _OPTIMIZER_TYPES:
        raise ValueError(f'{optimizer_type} is not an allowed type.  Allowed '
                         f'types are: {_OPTIMIZER_TYPES.keys()}')
    opts = _OPTIMIZER_TYPES[optimizer_type]
    for optimizer in opts:
        optimizer(copy)

    return circuits.Circuit(
        (op.transform_qubits(qubit_map) for op in copy.all_operations()),
        strategy=circuits.InsertStrategy.EARLIEST,
        device=new_device or copy.device)
