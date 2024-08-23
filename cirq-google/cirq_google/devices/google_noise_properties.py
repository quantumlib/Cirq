# Copyright 2022 The Cirq Developers
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


"""Class for representing noise on a Google device."""

import dataclasses
from functools import cached_property
from typing import Any, Dict, List, Sequence, Set, Type, TypeVar, Union

import numpy as np

import cirq, cirq_google
from cirq import _compat, devices
from cirq.devices import noise_utils
from cirq.transformers.heuristic_decompositions import gate_tabulation_math_utils


T = TypeVar('T')
V = TypeVar('V')


def _with_values(original: Dict[T, V], val: Union[V, Dict[T, V]]) -> Dict[T, V]:
    """Returns a copy of `original` using values from `val`.

    If val is a single value, all keys are mapped to that value. If val is a
    dict, the union of original and val is returned, using values from val for
    any conflicting keys.
    """
    if isinstance(val, dict):
        return {**original, **val}
    return {k: val for k in original}


@dataclasses.dataclass
class GoogleNoiseProperties(devices.SuperconductingQubitsNoiseProperties):
    """Noise-defining properties for a Google device.

    Inherited args:
        gate_times_ns: Dict[Type[`cirq.Gate`], float] of gate types to their
            duration on quantum hardware. Used with t(1|phi)_ns to specify
            thermal noise.
        t1_ns: Dict[`cirq.Qid`, float] of qubits to their T_1 time, in ns.
        tphi_ns: Dict[`cirq.Qid`, float] of qubits to their T_phi time, in ns.
        readout_errors: Dict[`cirq.Qid`, np.ndarray] of qubits to their readout
            errors in matrix form: [P(read |1> from |0>), P(read |0> from |1>)].
            Used to prepend amplitude damping errors to measurements.
        gate_pauli_errors: dict of `noise_utils.OpIdentifiers` (a gate and the
            qubits it targets) to the Pauli error for that operation. Used to
            construct depolarizing error. Keys in this dict must have defined
            qubits.
        validate: If True, verifies that t1 and tphi qubits sets match, and
            that all symmetric two-qubit gates have errors which are
            symmetric over the qubits they affect. Defaults to True.

    Additional args:
        fsim_errors: Dict[`noise_utils.OpIdentifier`, `cirq.PhasedFSimGate`] of
            gate types (potentially on specific qubits) to the PhasedFSim
            fix-up operation for that gate. Defaults to no-op for all gates.
    """

    fsim_errors: Dict[noise_utils.OpIdentifier, cirq.PhasedFSimGate] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        super().__post_init__()

        # validate two qubit gate errors.
        self._validate_symmetric_errors('fsim_errors')

    def __eq__(self, other):
        if not isinstance(other, GoogleNoiseProperties):
            return NotImplemented
        if set(self.readout_errors) != set(other.readout_errors):
            return False
        return all(
            [
                self.gate_times_ns == other.gate_times_ns,
                self.t1_ns == other.t1_ns,
                self.tphi_ns == other.tphi_ns,
                all(
                    np.allclose(self.readout_errors[q], other.readout_errors[q])
                    for q in self.readout_errors
                ),
                self.gate_pauli_errors == other.gate_pauli_errors,
                self.validate == other.validate,
                self.fsim_errors == other.fsim_errors,
            ]
        )

    def with_params(
        self,
        *,
        gate_times_ns: Union[None, float, Dict[Type['cirq.Gate'], float]] = None,
        t1_ns: Union[None, float, Dict['cirq.Qid', float]] = None,
        tphi_ns: Union[None, float, Dict['cirq.Qid', float]] = None,
        readout_errors: Union[None, Sequence[float], Dict['cirq.Qid', Sequence[float]]] = None,
        gate_pauli_errors: Union[
            None, float, Dict[Union[Type['cirq.Gate'], noise_utils.OpIdentifier], float]
        ] = None,
        fsim_errors: Union[
            None,
            'cirq.PhasedFSimGate',
            Dict[Union[Type['cirq.Gate'], noise_utils.OpIdentifier], 'cirq.PhasedFSimGate'],
        ] = None,
    ):
        """Returns a copy of this object with the given params overridden.

        This method supports partial replacement: each arg can accept a single
        value (which will replace all existing values) or a mapping (which will
        replace matching entries in the old object). Otherwise, all fields are
        the same as those used in the constructor.

        Args:
        gate_times_ns: float or Dict[Type[`cirq.Gate`], float].
        t1_ns: float or Dict[`cirq.Qid`, float].
        tphi_ns: float or Dict[`cirq.Qid`, float].
        readout_errors: Sequence or Dict[`cirq.Qid`, Sequence]. Converted to
            np.ndarray if not provided in that format.
        gate_pauli_errors: float or Dict[`cirq.OpIdentifier`, float].
            Dict key can also be Type[`cirq.Gate`]; this will apply the given
            error to all placements of that gate that appear in the original
            object.
        fsim_errors: `cirq.PhasedFSimGate` or Dict[`cirq.OpIdentifier`,
            `cirq.PhasedFSimGate`] Dict key can also be Type[`cirq.Gate`]; this
            will apply the given error to all placements of that gate that
            appear in the original object.

        """
        replace_args: Dict[str, Any] = {}
        if gate_times_ns is not None:
            replace_args['gate_times_ns'] = _with_values(self.gate_times_ns, gate_times_ns)
        if t1_ns is not None:
            replace_args['t1_ns'] = _with_values(self.t1_ns, t1_ns)
        if tphi_ns is not None:
            replace_args['tphi_ns'] = _with_values(self.tphi_ns, tphi_ns)
        if readout_errors is not None:
            if isinstance(readout_errors, dict):
                replace_args['readout_errors'] = _with_values(
                    self.readout_errors, {k: np.array(v) for k, v in readout_errors.items()}
                )
            else:
                replace_args['readout_errors'] = _with_values(
                    self.readout_errors, np.array(readout_errors)
                )
        if gate_pauli_errors is not None:
            if isinstance(gate_pauli_errors, dict):
                combined_pauli_errors: Dict[
                    Union[Type['cirq.Gate'], noise_utils.OpIdentifier], float
                ] = {}
                for op_id in self.gate_pauli_errors:
                    if op_id in gate_pauli_errors:
                        combined_pauli_errors[op_id] = gate_pauli_errors[op_id]
                    elif op_id.gate_type in gate_pauli_errors:
                        combined_pauli_errors[op_id] = gate_pauli_errors[op_id.gate_type]
                gate_pauli_errors = combined_pauli_errors
            replace_args['gate_pauli_errors'] = _with_values(
                self.gate_pauli_errors, gate_pauli_errors
            )
        if fsim_errors is not None:
            if isinstance(fsim_errors, dict):
                combined_fsim_errors: Dict[
                    Union[Type['cirq.Gate'], noise_utils.OpIdentifier], 'cirq.PhasedFSimGate'
                ] = {}
                for op_id in self.fsim_errors:
                    op_id_swapped = noise_utils.OpIdentifier(op_id.gate_type, *op_id.qubits[::-1])
                    if op_id in fsim_errors:
                        combined_fsim_errors[op_id] = fsim_errors[op_id]
                        combined_fsim_errors[op_id_swapped] = fsim_errors[op_id]
                    elif op_id_swapped in fsim_errors:
                        combined_fsim_errors[op_id] = fsim_errors[op_id_swapped]
                        combined_fsim_errors[op_id_swapped] = fsim_errors[op_id_swapped]
                    elif op_id.gate_type in fsim_errors:
                        combined_fsim_errors[op_id] = fsim_errors[op_id.gate_type]
                fsim_errors = combined_fsim_errors
            replace_args['fsim_errors'] = _with_values(self.fsim_errors, fsim_errors)
        return dataclasses.replace(self, **replace_args)

    @classmethod
    def single_qubit_gates(cls) -> Set[type]:
        return {cirq.ZPowGate, cirq.PhasedXZGate, cirq.MeasurementGate, cirq.ResetChannel}

    @classmethod
    def symmetric_two_qubit_gates(cls) -> Set[type]:
        return {
            cirq_google.SycamoreGate,
            cirq.FSimGate,
            cirq.PhasedFSimGate,
            cirq.ISwapPowGate,
            cirq.CZPowGate,
        }

    @classmethod
    def asymmetric_two_qubit_gates(cls) -> Set[type]:
        return set()

    @cached_property
    def _depolarizing_error(self) -> Dict[noise_utils.OpIdentifier, float]:
        depol_errors = super()._depolarizing_error

        def extract_entangling_error(match_id: noise_utils.OpIdentifier):
            """Gets the entangling error component of depol_errors[match_id]."""
            unitary_err = cirq.unitary(self.fsim_errors[match_id])
            fid = gate_tabulation_math_utils.unitary_entanglement_fidelity(unitary_err, np.eye(4))
            return 1 - fid

        # Subtract entangling angle error.
        for op_id in depol_errors:
            if op_id.gate_type not in self.two_qubit_gates():
                continue
            if op_id in self.fsim_errors:
                depol_errors[op_id] -= extract_entangling_error(op_id)
                continue
            # Find the closest matching supertype, if one is provided.
            # Gateset has similar behavior, but cannot be used here
            # because depol_errors is a dict, not a set.
            match_id = None
            candidate_parents = [
                parent_id for parent_id in self.fsim_errors if op_id.is_proper_subtype_of(parent_id)
            ]
            for parent_id in candidate_parents:
                if match_id is None or parent_id.is_proper_subtype_of(match_id):
                    match_id = parent_id
            if match_id is not None:
                depol_errors[op_id] -= extract_entangling_error(match_id)

        return depol_errors

    def build_noise_models(self) -> List['cirq.NoiseModel']:
        """Construct all NoiseModels associated with NoiseProperties."""
        noise_models = super().build_noise_models()

        # Insert entangling gate coherent errors after thermal error.
        if self.fsim_errors:
            fsim_ops = {op_id: gate.on(*op_id.qubits) for op_id, gate in self.fsim_errors.items()}
            noise_models.insert(1, devices.InsertionNoiseModel(ops_added=fsim_ops))

        return noise_models

    def __repr__(self):
        gate_times_repr = ', '.join(
            f'{key.__module__}.{key.__qualname__}: {val}' for key, val in self.gate_times_ns.items()
        )
        args = [
            f'gate_times_ns={{{gate_times_repr}}}',
            f't1_ns={self.t1_ns!r}',
            f'tphi_ns={self.tphi_ns!r}',
            f'readout_errors={_compat.proper_repr(self.readout_errors)}',
            f'gate_pauli_errors={self.gate_pauli_errors!r}',
            f'fsim_errors={self.fsim_errors!r}',
            f'validate={self.validate!r}',
        ]
        arglines = ",\n".join(args)
        return f'cirq_google.GoogleNoiseProperties({arglines})'

    def _json_dict_(self):
        storage_gate_times = {
            cirq.json_cirq_type(key): val for key, val in self.gate_times_ns.items()
        }
        return {
            # JSON requires mappings to have keys of basic types.
            # Pairs must be sorted to ensure consistent serialization.
            'gate_times_ns': tuple(storage_gate_times.items()),
            't1_ns': tuple(self.t1_ns.items()),
            'tphi_ns': tuple(self.tphi_ns.items()),
            'readout_errors': tuple((k, v.tolist()) for k, v in self.readout_errors.items()),
            'gate_pauli_errors': tuple(self.gate_pauli_errors.items()),
            'fsim_errors': tuple(self.fsim_errors.items()),
            'validate': self.validate,
        }

    @classmethod
    def _from_json_dict_(
        cls,
        gate_times_ns,
        t1_ns,
        tphi_ns,
        readout_errors,
        gate_pauli_errors,
        fsim_errors,
        validate,
        **kwargs,
    ):
        gate_type_times = {cirq.cirq_type_from_json(gate): val for gate, val in gate_times_ns}
        # Known false positive: https://github.com/PyCQA/pylint/issues/5857
        return GoogleNoiseProperties(  # pylint: disable=unexpected-keyword-arg
            gate_times_ns=gate_type_times,
            t1_ns=dict(t1_ns),
            tphi_ns=dict(tphi_ns),
            readout_errors={k: np.array(v) for k, v in readout_errors},
            gate_pauli_errors=dict(gate_pauli_errors),
            fsim_errors=dict(fsim_errors),
            validate=validate,
        )


class NoiseModelFromGoogleNoiseProperties(devices.NoiseModelFromNoiseProperties):
    """A noise model defined from noise properties of a Google device."""

    def is_virtual(self, op: cirq.Operation) -> bool:
        return isinstance(op.gate, cirq.ZPowGate) and cirq_google.PhysicalZTag not in op.tags

    # noisy_moments is implemented by the superclass.
