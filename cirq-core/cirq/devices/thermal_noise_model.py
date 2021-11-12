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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union
from scipy.linalg import expm
import numpy as np

from cirq import devices, ops, protocols, qis
from cirq.devices.noise_utils import (
    PHYSICAL_GATE_TAG,
)

if TYPE_CHECKING:
    import cirq


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


def _validate_rates(qubit_dims: Dict['cirq.Qid', int], rates: Dict['cirq.Qid', np.ndarray]) -> None:
    """Check all rate matrices are square and of appropriate dimension.

    We check rates are positive in the class validator.
    """
    if set(qubit_dims) != set(rates):
        raise ValueError('qubits for rates inconsistent with those through qubit_dims')
    for q in rates:
        if rates[q].shape != (qubit_dims[q], qubit_dims[q]):
            raise ValueError(
                f'Invalid shape for rate matrix: should be ({qubit_dims[q]}, {qubit_dims[q]}), '
                f'but got {rates[q].shape}'
            )


@dataclass
class ThermalNoiseModel(devices.NoiseModel):
    """NoiseModel representing simulated thermalization of a qubit.

    This model is designed for qubits which use energy levels as their states.
    "Heating" and "cooling" here are used to refer to environmental noise which
    transitions a qubit to higher or lower energy levels, respectively.
    """

    def __init__(
        self,
        qubit_dims: Dict['cirq.Qid', int],
        gate_durations_ns: Dict[type, float],
        heat_rate_GHz: Union[float, Dict['cirq.Qid', float], None] = None,
        cool_rate_GHz: Union[float, Dict['cirq.Qid', float], None] = None,
        dephase_rate_GHz: Union[float, Dict['cirq.Qid', float], None] = None,
        require_physical_tag: bool = True,
        skip_measurements: bool = True,
    ):
        """Construct a ThermalNoiseModel data object.

        Required Args:
            qubit_dims: Dimension for all qubits in the system.
                        Currently only supports dimension=2 (qubits, not qudits)
        Optional Args:
            heat_rate_GHz: single number (units GHz) specifying heating rate,
                        either per qubit, or global value for all.
                        Given a rate gh, the Lindblad op will be sqrt(gh)*a^dag
                        (where a is annihilation),
                        so that the heating Lindbldian is
                        gh(a^dag • a - 0.5{a*a^dag, •}).
            cool_rate_GHz: single number (units GHz) specifying cooling rate,
                        either per qubit, or global value for all.
                        Given a rate gc, the Lindblad op will be sqrt(gc)*a
                        so that the cooling Lindbldian is
                        gc(a • a^dag - 0.5{n, •})
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
        qubits = set(qubit_dims)
        rate_dict = {}

        def _as_rate_dict(
            rate_or_dict: Optional[Union[float, Dict['cirq.Qid', float]]]
        ) -> Dict['cirq.Qid', float]:
            # Convert float or None input into dictionary form. Make sure no
            # qubits are missing from dictionary input.
            if rate_or_dict is None:
                return {qb: 0.0 for qb in qubits}
            elif isinstance(rate_or_dict, dict):
                out = rate_or_dict.copy()
                for qb in qubits:
                    if qb not in rate_or_dict:
                        out[qb] = 0.0
                return out
            else:
                return {qb: rate_or_dict for qb in qubits}

        heat_rate_GHz = _as_rate_dict(heat_rate_GHz)
        cool_rate_GHz = _as_rate_dict(cool_rate_GHz)
        dephase_rate_GHz = _as_rate_dict(dephase_rate_GHz)

        for q, dim in qubit_dims.items():
            gamma_h = heat_rate_GHz[q]
            gamma_c = cool_rate_GHz[q]
            gamma_phi = dephase_rate_GHz[q]

            rate_dict[q] = _decoherence_matrix(gamma_c, gamma_phi, gamma_h, dim)

        _validate_rates(qubit_dims, rate_dict)
        self.gate_durations_ns: Dict[type, float] = gate_durations_ns
        self.rate_matrix_GHz: Dict['cirq.Qid', np.ndarray] = rate_dict
        self.require_physical_tag: bool = require_physical_tag
        self.skip_measurements: bool = skip_measurements

    def noisy_moment(
        self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        noise_ops: List['cirq.Operation'] = []
        moment_ns: float = 0
        for op in moment:
            op_duration: Optional[float] = None
            for key, duration in self.gate_durations_ns.items():
                if not issubclass(type(op.gate), key):
                    continue  # gate type doesn't match
                # TODO: remove assumption of same time across qubits
                # if len(key) > 1 and op_data[:1] != key[:1]:
                #     continue  # qubits don't match
                op_duration = duration
                break
            if op_duration is None:
                if not isinstance(op.gate, ops.WaitGate):
                    continue
                # special case for wait gates if not predefined
                op_duration = op.gate.duration.total_nanos()
            moment_ns = max(moment_ns, op_duration)

        if moment_ns == 0:
            return [moment]

        for qubit in system_qubits:
            qubit_op = moment.operation_at(qubit)
            if qubit_op is None:
                continue
            if self.skip_measurements and protocols.is_measurement(qubit_op):
                continue
            if self.require_physical_tag and PHYSICAL_GATE_TAG not in qubit_op.tags:
                # Only non-virtual gates get noise applied.
                continue
            rates = self.rate_matrix_GHz[qubit] * moment_ns
            num_op = np.diag(np.sqrt(np.diag(rates)))
            annihilation = np.sqrt(np.triu(rates, 1))
            creation = np.sqrt(np.triu(rates.T, 1)).T
            # Lindbladian with three Lindblad ops for the three processes
            # Note: 'time' parameter already specified implicitly through rates
            L = _lindbladian(annihilation) + _lindbladian(creation) + 2 * _lindbladian(num_op)
            superop = expm(L.real)
            kraus_ops = qis.superoperator_to_kraus(superop)
            noise_ops.append(ops.KrausChannel(kraus_ops).on(qubit))
        if not noise_ops:
            return [moment]
        return [moment, ops.Moment(noise_ops)]
