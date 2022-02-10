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

import dataclasses
import functools
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import scipy.linalg

from cirq import devices, ops, protocols, qis
from cirq._import import LazyLoader
from cirq.devices.noise_utils import (
    PHYSICAL_GATE_TAG,
)

if TYPE_CHECKING:
    import cirq

moment_module = LazyLoader("moment_module", globals(), "cirq.circuits.moment")


def _left_mul(mat: np.ndarray) -> np.ndarray:
    """Superoperator associated with left multiplication by a square matrix."""
    mat = np.asarray(mat)
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError(
            f'_left_mul only accepts square matrices, but input matrix has shape {mat.shape}.'
        )
    dim = mat.shape[-1]

    return np.kron(mat, np.eye(dim))


def _right_mul(mat: np.ndarray) -> np.ndarray:
    """Superoperator associated with right multiplication by a square matrix."""
    mat = np.asarray(mat)
    if mat.shape[-1] != mat.shape[-2]:
        raise ValueError(
            f'_right_mul only accepts square matrices, but input matrix has shape {mat.shape}.'
        )
    dim = mat.shape[-1]

    return np.kron(np.eye(dim), np.swapaxes(mat, -2, -1))


def _lindbladian(left_op: np.ndarray) -> np.ndarray:
    r"""Superoperator representing a Lindbladian.

    The Lindbladian generated by a single operator A is the superoperator

        $$
        L(\rho) = A \rho A^\dagger - 0.5 (A^\dagger A \rho + \rho A^\dagger A)
        $$

    Args:
        left_op: The operator acting on the left in the Lindbladian (A above).

    Returns:
        Superoperator corresponding to the Lindbladian.
    """
    left_op = np.asarray(left_op)
    right_op = left_op.conj().T
    square = right_op @ left_op
    out = _left_mul(left_op) @ _right_mul(right_op)
    out -= 0.5 * (_left_mul(square) + _right_mul(square))
    return out


@functools.lru_cache(maxsize=256)
def _kraus_ops_from_rates(
    flat_rates: Tuple[float, ...], shape: Tuple[int, int]
) -> Sequence[np.ndarray]:
    """Generate kraus operators from an array of rates.

    Args:
        flat_rates: A tuple of rates, flattened from a numpy array with:
            flat_rates = tuple(rates.reshape(-1))
            This format is necessary to support caching of inputs.
        shape: The shape of flat_rates prior to flattening.
    """
    rates = np.array(flat_rates).reshape(shape)
    num_op = np.diag(np.sqrt(np.diag(rates)))
    annihilation = np.sqrt(np.triu(rates, 1))
    creation = np.sqrt(np.triu(rates.T, 1)).T
    # Lindbladian with three Lindblad ops for the three processes
    # Note: 'time' parameter already specified implicitly through rates
    L = _lindbladian(annihilation) + _lindbladian(creation) + 2 * _lindbladian(num_op)
    superop = scipy.linalg.expm(L.real)
    return qis.superoperator_to_kraus(superop)


def _decoherence_matrix(
    cool_rate: float, dephase_rate: float, heat_rate: float = 0.0, dim: int = 2
) -> np.ndarray:
    """Construct a rate matrix associated with decay and dephasing.

    The units of the matrix match the units of the rates specified.
    This matrix can be used to construct a noise channel after rescaling
    by an idling time (to make it dimensionless).

    Args:
        cool_rate: Decay rate of the system, usually 1 / T_1
        dephase_rate: Static dephasing rate of the system, usually 1 / T_phi
        heat_rate: Heating rate of the system (default 0).
        dim: Number of energy levels to include (default 2).

    Returns:
        np.ndarray rate matrix for decay and dephasing.
    """
    # Heating (related to a^dag)
    rate_matrix = np.diag(np.arange(1, dim) * heat_rate, 1).T.astype(float)
    # Cooling (related to a)
    rate_matrix += np.diag(np.arange(1, dim) * cool_rate, 1)
    # Dephasing (related to n=a^dag * a)
    # We specify i^2 since we take square root to get the Lindblad op later.
    rate_matrix += np.diag(dephase_rate * np.arange(dim) ** 2)
    return rate_matrix


def _as_rate_dict(
    rate_or_dict: Optional[Union[float, Dict['cirq.Qid', float]]],
    qubits: Set['cirq.Qid'],
) -> Dict['cirq.Qid', float]:
    """Convert float or None input into dictionary form.

    This method also ensures that no qubits are missing from dictionary keys.
    """
    if rate_or_dict is None:
        return {q: 0.0 for q in qubits}
    elif isinstance(rate_or_dict, dict):
        return {**{q: 0.0 for q in qubits}, **rate_or_dict}
    else:
        return {q: rate_or_dict for q in qubits}


def _validate_rates(qubits: Set['cirq.Qid'], rates: Dict['cirq.Qid', np.ndarray]) -> None:
    """Check all rate matrices are square and of appropriate dimension.

    We check rates are positive in the class validator.
    """
    if qubits != set(rates):
        raise ValueError('qubits for rates inconsistent with those through qubit_dims')
    for q in rates:
        if rates[q].shape != (q.dimension, q.dimension):
            raise ValueError(
                f'Invalid shape for rate matrix: should be ({q.dimension}, {q.dimension}), '
                f'but got {rates[q].shape}'
            )


@dataclasses.dataclass
class ThermalNoiseModel(devices.NoiseModel):
    """NoiseModel representing simulated thermalization of a qubit.

    This model is designed for qubits which use energy levels as their states. "Heating" and
    "cooling" here are used to refer to environmental noise which transitions a qubit to higher or
    lower energy levels, respectively.
    """

    def __init__(
        self,
        qubits: Set['cirq.Qid'],
        gate_durations_ns: Dict[type, float],
        heat_rate_GHz: Union[float, Dict['cirq.Qid', float], None] = None,
        cool_rate_GHz: Union[float, Dict['cirq.Qid', float], None] = None,
        dephase_rate_GHz: Union[float, Dict['cirq.Qid', float], None] = None,
        require_physical_tag: bool = True,
        skip_measurements: bool = True,
    ):
        """Construct a ThermalNoiseModel data object.

        Required Args:
            qubits: Set of all qubits in the system.
            gate_durations_ns: Map of gate types to their duration in
                nanoseconds. These values will override default values for
                gate duration, if any (e.g. WaitGate).
        Optional Args:
            heat_rate_GHz: single number (units GHz) specifying heating rate,
                either per qubit, or global value for all.
                Given a rate gh, the Lindblad op will be sqrt(gh)*a^dag
                (where a is annihilation), so that the heating Lindbldian is
                gh(a^dag • a - 0.5{a*a^dag, •}).
            cool_rate_GHz: single number (units GHz) specifying cooling rate,
                either per qubit, or global value for all.
                Given a rate gc, the Lindblad op will be sqrt(gc)*a
                so that the cooling Lindbldian is gc(a • a^dag - 0.5{n, •})
                This number is equivalent to 1/T1.
            dephase_rate_GHz: single number (units GHz) specifying dephasing
                rate, either per qubit, or global value for all.
                Given a rate gd, Lindblad op will be sqrt(2*gd)*n where
                n = a^dag * a, so that the dephasing Lindbldian is
                2 * gd * (n • n - 0.5{n^2, •}).
                This number is equivalent to 1/Tphi.
            require_physical_tag: whether to only apply noise to operations
                tagged with PHYSICAL_GATE_TAG.
            skip_measurements: whether to skip applying noise to measurements.

        Returns:
            The ThermalNoiseModel with specified parameters.
        """
        rate_dict = {}

        heat_rate_GHz = _as_rate_dict(heat_rate_GHz, qubits)
        cool_rate_GHz = _as_rate_dict(cool_rate_GHz, qubits)
        dephase_rate_GHz = _as_rate_dict(dephase_rate_GHz, qubits)

        for q in qubits:
            gamma_h = heat_rate_GHz[q]
            gamma_c = cool_rate_GHz[q]
            gamma_phi = dephase_rate_GHz[q]

            rate_dict[q] = _decoherence_matrix(gamma_c, gamma_phi, gamma_h, q.dimension)

        _validate_rates(qubits, rate_dict)
        self.gate_durations_ns: Dict[type, float] = gate_durations_ns
        self.rate_matrix_GHz: Dict['cirq.Qid', np.ndarray] = rate_dict
        self.require_physical_tag: bool = require_physical_tag
        self.skip_measurements: bool = skip_measurements

    def noisy_moment(
        self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        if not moment.operations:
            return [moment]
        if self.require_physical_tag:
            physical_ops = [PHYSICAL_GATE_TAG in op.tags for op in moment]
            if any(physical_ops):
                if not all(physical_ops):
                    raise ValueError(
                        "Moments are expected to be all physical or all virtual ops, "
                        f"but found {moment.operations}"
                    )
            else:
                # Only moments with physical operations should have noise.
                return [moment]

        noise_ops: List['cirq.Operation'] = []
        # Some devices (including Google hardware) require that all gates have
        # the same duration, but this does not. Instead, each moment is assumed
        # to be as long as the longest gate it contains.
        moment_ns: float = 0
        for op in moment:
            op_duration: Optional[float] = None
            for key, duration in self.gate_durations_ns.items():
                if not issubclass(type(op.gate), key):
                    continue  # gate type doesn't match
                # TODO: remove assumption of same time across qubits
                op_duration = duration
                break
            if op_duration is None and isinstance(op.gate, ops.WaitGate):
                # special case for wait gates if not predefined
                op_duration = op.gate.duration.total_nanos()
            if op_duration is not None:
                moment_ns = max(moment_ns, op_duration)

        if moment_ns == 0:
            return [moment]

        for qubit in system_qubits:
            qubit_op = moment.operation_at(qubit)
            if self.skip_measurements and protocols.is_measurement(qubit_op):
                continue
            rates = self.rate_matrix_GHz[qubit] * moment_ns
            kraus_ops = _kraus_ops_from_rates(tuple(rates.reshape(-1)), rates.shape)
            noise_ops.append(ops.KrausChannel(kraus_ops).on(qubit))
        if not noise_ops:
            return [moment]
        return [moment, moment_module.Moment(noise_ops)]
