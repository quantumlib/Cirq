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
from functools import lru_cache
from typing import Callable, cast, List, Optional, TYPE_CHECKING

import numpy as np

from cirq import circuits, devices, optimizers, protocols
from cirq.google import ops as cg_ops
from cirq.google.optimizers import (
    convert_to_xmon_gates,
    ConvertToSycamoreGates,
    ConvertToSqrtIswapGates,
    gate_product_tabulation,
    GateTabulation,
)

if TYPE_CHECKING:
    import cirq


def _get_common_cleanup_optimizers(tolerance: float
                                  ) -> List[Callable[['cirq.Circuit'], None]]:
    return [
        optimizers.EjectPhasedPaulis(tolerance=tolerance).optimize_circuit,
        optimizers.EjectZ(tolerance=tolerance).optimize_circuit,
        optimizers.DropNegligible(tolerance=tolerance).optimize_circuit,
    ]


def _get_xmon_optimizers(tolerance: float, tabulation: Optional[GateTabulation]
                        ) -> List[Callable[['cirq.Circuit'], None]]:
    if tabulation is not None:
        # coverage: ignore
        raise ValueError("Gate tabulation not supported for xmon")

    return [
        convert_to_xmon_gates.ConvertToXmonGates().optimize_circuit,
        optimizers.MergeInteractions(tolerance=tolerance,
                                     allow_partial_czs=False).optimize_circuit,
        lambda c: optimizers.merge_single_qubit_gates_into_phxz(c, tolerance),
        *_get_common_cleanup_optimizers(tolerance=tolerance),
    ]


def _get_xmon_optimizers_part_cz(tolerance: float,
                                 tabulation: Optional[GateTabulation]
                                ) -> List[Callable[['cirq.Circuit'], None]]:
    if tabulation is not None:
        # coverage: ignore
        raise ValueError("Gate tabulation not supported for xmon")
    return [
        convert_to_xmon_gates.ConvertToXmonGates().optimize_circuit,
        optimizers.MergeInteractions(tolerance=tolerance,
                                     allow_partial_czs=True).optimize_circuit,
        lambda c: optimizers.merge_single_qubit_gates_into_phxz(c, tolerance),
        *_get_common_cleanup_optimizers(tolerance=tolerance),
    ]


def _get_sycamore_optimizers(tolerance: float,
                             tabulation: Optional[GateTabulation]
                            ) -> List[Callable[['cirq.Circuit'], None]]:
    return [
        ConvertToSycamoreGates(tabulation=tabulation).optimize_circuit,
        lambda c: optimizers.merge_single_qubit_gates_into_phxz(c, tolerance),
        *_get_common_cleanup_optimizers(tolerance=tolerance),
    ]


def _get_sqrt_iswap_optimizers(tolerance: float,
                               tabulation: Optional[GateTabulation]
                              ) -> List[Callable[['cirq.Circuit'], None]]:
    if tabulation is not None:
        # coverage: ignore
        raise ValueError("Gate tabulation not supported for sqrt_iswap")
    return [
        ConvertToSqrtIswapGates().optimize_circuit,
        lambda c: optimizers.merge_single_qubit_gates_into_phxz(c, tolerance),
        *_get_common_cleanup_optimizers(tolerance=tolerance),
    ]


_OPTIMIZER_TYPES = {
    'xmon': _get_xmon_optimizers,
    'xmon_partial_cz': _get_xmon_optimizers_part_cz,
    'sqrt_iswap': _get_sqrt_iswap_optimizers,
    'sycamore': _get_sycamore_optimizers,
}


@lru_cache()
def _gate_product_tabulation_cached(optimizer_type: str,
                                    tabulation_resolution: float
                                   ) -> GateTabulation:
    random_state = np.random.RandomState(51)
    if optimizer_type == 'sycamore':
        return gate_product_tabulation(protocols.unitary(cg_ops.SYC),
                                       tabulation_resolution,
                                       random_state=random_state)
    else:
        raise NotImplementedError(
            f"Gate tabulation not supported for {optimizer_type}")


def optimized_for_sycamore(
        circuit: 'cirq.Circuit',
        *,
        new_device: Optional['cirq.google.XmonDevice'] = None,
        qubit_map: Callable[['cirq.Qid'], devices.GridQubit] = lambda e: cast(
            devices.GridQubit, e),
        optimizer_type: str = 'sqrt_iswap',
        tolerance: float = 1e-5,
        tabulation_resolution: Optional[float] = None,
) -> 'cirq.Circuit':
    """Optimizes a circuit for Google devices.

    Uses a set of optimizers that will compile to the proper gateset for the
    device (xmon, sqrt_iswap, or sycamore gates) and then use optimizers to
    compress the gate depth down as much as is easily algorithmically possible
    by merging rotations, ejecting Z gates, etc.

    Args:
        circuit: The circuit to optimize.
        new_device: The device the optimized circuit should be targeted at. If
            set to None, the circuit's current device is used.
        qubit_map: Transforms the qubits (e.g. so that they are GridQubits).
        optimizer_type: A string defining the optimizations to apply.
            Possible values are  'xmon', 'xmon_partial_cz', 'sqrt_iswap',
            'sycamore'
        tolerance: The tolerance passed to the various circuit optimization
            passes.
        tabulation_resolution: If provided, compute a gateset tabulation
            with the specified resolution and use it to approximately
            compile arbitrary two-qubit gates for which an analytic compilation
            is not known.
    Returns:
        The optimized circuit.
    """
    copy = circuit.copy()
    if optimizer_type not in _OPTIMIZER_TYPES:
        raise ValueError(f'{optimizer_type} is not an allowed type.  Allowed '
                         f'types are: {_OPTIMIZER_TYPES.keys()}')

    tabulation: Optional[GateTabulation] = None
    if tabulation_resolution is not None:
        tabulation = _gate_product_tabulation_cached(optimizer_type,
                                                     tabulation_resolution)

    opts = _OPTIMIZER_TYPES[optimizer_type](tolerance=tolerance,
                                            tabulation=tabulation)
    for optimizer in opts:
        optimizer(copy)

    return circuits.Circuit(
        (op.transform_qubits(qubit_map) for op in copy.all_operations()),
        strategy=circuits.InsertStrategy.EARLIEST,
        device=new_device or copy.device)
