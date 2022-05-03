# Copyright 2020 The Cirq Developers
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

import dataclasses
import numbers
from typing import Union, Iterable, Dict, TYPE_CHECKING, ItemsView, Tuple, FrozenSet

import sympy

from cirq import ops, value, protocols

if TYPE_CHECKING:
    import cirq
    from cirq.value.product_state import _NamedOneQubitState


@dataclasses.dataclass(frozen=True)
class InitObsSetting:
    """A pair of initial state and observable.

    Usually, given a circuit you want to iterate through many
    InitObsSettings to vary the initial state preparation and output
    observable.
    """

    init_state: value.ProductState
    observable: ops.PauliString

    def __post_init__(self):
        # Special validation for this dataclass.
        init_qs = self.init_state.qubits
        obs_qs = self.observable.qubits
        if set(obs_qs) > set(init_qs):
            raise ValueError(
                "`observable`'s qubits should be a subset of those "
                "found in `init_state`. "
                "observable qubits: {}. init_state qubits: {}".format(obs_qs, init_qs)
            )

    def __str__(self):
        return f'{self.init_state} → {self.observable}'

    def __repr__(self):
        return (
            f'cirq.work.InitObsSetting('
            f'init_state={self.init_state!r}, '
            f'observable={self.observable!r})'
        )

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)


def _max_weight_observable(observables: Iterable[ops.PauliString]) -> Union[None, ops.PauliString]:
    """Create a new observable that is compatible with all input observables
    and has the maximum non-identity elements.

    The returned PauliString is constructed by taking the non-identity
    single-qubit Pauli at each qubit position.

    This function will return `None` if the input observables do not share a
    tensor product basis.

    For example, the _max_weight_observable of ["XI", "IZ"] is "XZ". Asking for
    the max weight observable of something like ["XI", "ZI"] will return None.

    The returned value need not actually be present in the input observables.
    Coefficients from input observables will be dropped.
    """
    qubit_pauli_map: Dict[ops.Qid, ops.Pauli] = {}
    for observable in observables:
        for qubit, pauli in observable.items():
            if qubit in qubit_pauli_map:
                if qubit_pauli_map[qubit] != pauli:
                    return None
            else:
                qubit_pauli_map[qubit] = pauli
    return ops.PauliString(qubit_pauli_map)


def _max_weight_state(states: Iterable[value.ProductState]) -> Union[None, value.ProductState]:
    """Create a new state that is compatible with all input states
    and has the maximum weight.

    The returned TensorProductState is constructed by taking the
    single-qubit state at each qubit position.

    This function will return `None` if the input states are not compatible

    For example, the max_weight_state of [+X(0), -Z(1)] is
    "+X(0) * -Z(1)". Asking for the max weight state of something like
    [+X(0), +Z(0)] will return None.
    """
    qubit_state_map: Dict[ops.Qid, _NamedOneQubitState] = {}
    for state in states:
        for qubit, named_state in state:
            if qubit in qubit_state_map:
                if qubit_state_map[qubit] != named_state:
                    return None
            else:
                qubit_state_map[qubit] = named_state
    return value.ProductState(qubit_state_map)


def zeros_state(qubits: Iterable['cirq.Qid']):
    """Return the ProductState that is |00..00> on all qubits."""
    return value.ProductState({q: value.KET_ZERO for q in qubits})


def observables_to_settings(
    observables: Iterable['cirq.PauliString'], qubits: Iterable['cirq.Qid']
) -> Iterable[InitObsSetting]:
    """Transform an observable to an InitObsSetting initialized in the
    all-zeros state.
    """
    for observable in observables:
        yield InitObsSetting(init_state=zeros_state(qubits), observable=observable)


def _fix_precision(val: Union[value.Scalar, sympy.Expr], precision) -> Union[int, Tuple[int, int]]:
    """Convert floating point or complex numbers to (implicitly) fixed point
    integers. Complex numbers will return fixed-point (real, imag) tuples.

    Circuit parameters can be complex but we also need to use them as
    dictionary keys. We secretly use these fixed-precision integers.
    """
    if isinstance(val, sympy.Expr):
        raise ValueError(f'Cannot convert {val} to fixed precision in observable settings')
    if isinstance(val, (complex, numbers.Complex)):
        return int(val.real * precision), int(val.imag * precision)
    return int(val * precision)


def _hashable_param(
    param_tuples: ItemsView[Union[str, sympy.Expr], Union[value.Scalar, sympy.Expr]], precision=1e7
) -> FrozenSet[Tuple[str, Union[int, Tuple[int, int]]]]:
    """Hash circuit parameters using fixed precision.

    Circuit parameters can be complex but we also need to use them as
    dictionary keys. We secretly use these fixed-precision integers.
    """
    return frozenset(
        (k, _fix_precision(v, precision)) for k, v in param_tuples if isinstance(k, str)
    )


@dataclasses.dataclass(frozen=True)
class _MeasurementSpec:
    """An encapsulation of all the specifications for one run of a
    quantum processor.

    This includes the maximal input-output setting (which may result in many
    observables being measured if they are consistent with `max_setting`) and
    a set of circuit parameters if the circuit is parameterized.
    """

    max_setting: InitObsSetting
    circuit_params: Dict[Union[str, sympy.Expr], Union[value.Scalar, sympy.Expr]]

    def __hash__(self):
        return hash((self.max_setting, _hashable_param(self.circuit_params.items())))

    def __repr__(self):
        return (
            f'cirq.work._MeasurementSpec(max_setting={self.max_setting!r}, '
            f'circuit_params={self.circuit_params!r})'
        )

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)
