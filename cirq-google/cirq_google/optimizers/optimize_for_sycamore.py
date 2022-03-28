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
from typing import Callable, cast, Optional, TYPE_CHECKING

import numpy as np

import cirq
from cirq_google import ops as cg_ops
from cirq_google.transformers.target_gatesets import sycamore_gateset

if TYPE_CHECKING:
    import cirq_google


_TARGET_GATESETS = {
    'sqrt_iswap': lambda atol, _: cirq.SqrtIswapTargetGateset(atol=atol),
    'sycamore': lambda atol, tabulation: sycamore_gateset.SycamoreTargetGateset(
        atol=atol, tabulation=tabulation
    ),
    'xmon': lambda atol, _: cirq.CZTargetGateset(atol=atol),
    'xmon_partial_cz': lambda atol, _: cirq.CZTargetGateset(atol=atol, allow_partial_czs=True),
}


@lru_cache()
def _gate_product_tabulation_cached(
    optimizer_type: str, tabulation_resolution: float
) -> cirq.TwoQubitGateTabulation:
    random_state = np.random.RandomState(51)
    if optimizer_type == 'sycamore':
        return cirq.two_qubit_gate_product_tabulation(
            cirq.unitary(cg_ops.SYC), tabulation_resolution, random_state=random_state
        )
    else:
        raise NotImplementedError(f"Two qubit gate tabulation not supported for {optimizer_type}")


@cirq._compat.deprecated_parameter(
    deadline='v0.15',
    fix=cirq.circuits.circuit._DEVICE_DEP_MESSAGE,
    parameter_desc='new_device',
    match=lambda args, kwargs: 'new_device' in kwargs,
)
def optimized_for_sycamore(
    circuit: cirq.Circuit,
    *,
    new_device: Optional['cirq_google.XmonDevice'] = None,
    qubit_map: Callable[[cirq.Qid], cirq.GridQubit] = lambda e: cast(cirq.GridQubit, e),
    optimizer_type: str = 'sqrt_iswap',
    tolerance: float = 1e-5,
    tabulation_resolution: Optional[float] = None,
) -> cirq.Circuit:
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

    Raises:
        ValueError: If the `optimizer_type` is not a supported type.
    """
    copy = circuit.copy()
    if optimizer_type not in _TARGET_GATESETS:
        raise ValueError(
            f'{optimizer_type} is not an allowed type.  Allowed '
            f'types are: {_TARGET_GATESETS.keys()}'
        )

    tabulation: Optional[cirq.TwoQubitGateTabulation] = None
    if tabulation_resolution is not None:
        tabulation = _gate_product_tabulation_cached(optimizer_type, tabulation_resolution)

    if optimizer_type in _TARGET_GATESETS:
        copy = cirq.optimize_for_target_gateset(
            circuit,
            gateset=_TARGET_GATESETS[optimizer_type](tolerance, tabulation),
            context=cirq.TransformerContext(deep=True),
        )
    copy = cirq.merge_single_qubit_gates_to_phxz(copy, atol=tolerance)
    copy = cirq.eject_phased_paulis(copy, atol=tolerance)
    copy = cirq.eject_z(copy, atol=tolerance)
    copy = cirq.drop_negligible_operations(copy, atol=tolerance)

    ret = cirq.Circuit(
        (op.transform_qubits(qubit_map) for op in copy.all_operations()),
        strategy=cirq.InsertStrategy.EARLIEST,
    )
    ret._device = new_device or copy._device
    return ret
