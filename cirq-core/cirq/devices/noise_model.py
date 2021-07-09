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

from typing import Any, Dict, Sequence, TYPE_CHECKING, Union, Callable

from cirq import ops, protocols, value
from cirq._doc import document

if TYPE_CHECKING:
    from typing import Iterable
    import cirq


class NoiseModel(metaclass=value.ABCMetaImplementAnyOneOf):
    """Replaces operations and moments with noisy counterparts.

    A child class must override *at least one* of the following three methods:

        noisy_moments
        noisy_moment
        noisy_operation

    The methods that are not overridden will be implemented in terms of the ones
    that are.

    Simulators told to use a noise model will use these methods in order to
    dynamically rewrite the program they are simulating.
    """

    @classmethod
    def from_noise_model_like(cls, noise: 'cirq.NOISE_MODEL_LIKE') -> 'cirq.NoiseModel':
        """Transforms an object into a noise model if umambiguously possible.

        Args:
            noise: ``None``, a ``cirq.NoiseModel``, or a single qubit operation.

        Returns:
            ``cirq.NO_NOISE`` when given ``None``,
            ``cirq.ConstantQubitNoiseModel(gate)`` when given a single qubit
            gate, or the given value if it is already a ``cirq.NoiseModel``.

        Raises:
            TypeError: The input is not a ``cirq.NOISE_MODE_LIKE``.
        """
        if noise is None:
            return NO_NOISE
        if isinstance(noise, NoiseModel):
            return noise
        if isinstance(noise, ops.Gate):
            if noise.num_qubits() != 1:
                raise ValueError(
                    'Multi-qubit gates cannot be implicitly wrapped into a '
                    'noise model. Please use a single qubit gate (which will '
                    'be wrapped with `cirq.ConstantQubitNoiseModel`) or '
                    'an instance of `cirq.NoiseModel`.'
                )
            return ConstantQubitNoiseModel(noise)

        raise TypeError(
            'Expected a NOISE_MODEL_LIKE (None, a cirq.NoiseModel, '
            'or a single qubit gate). Got {!r}'.format(noise)
        )

    def is_virtual_moment(self, moment: 'cirq.Moment') -> bool:
        """Returns true iff the given moment is non-empty and all of its
        operations are virtual.

        Moments for which this method returns True should not have additional
        noise applied to them.

        Args:
            moment: ``cirq.Moment`` to check for non-virtual operations.

        Returns:
            True if "moment" is non-empty and all operations in "moment" are
            virtual; false otherwise.
        """
        if not moment.operations:
            return False
        return all(ops.VirtualTag() in op.tags for op in moment)

    def _noisy_moments_impl_moment(
        self, moments: 'Iterable[cirq.Moment]', system_qubits: Sequence['cirq.Qid']
    ) -> Sequence['cirq.OP_TREE']:
        result = []
        for moment in moments:
            result.append(self.noisy_moment(moment, system_qubits))
        return result

    def _noisy_moments_impl_operation(
        self, moments: 'Iterable[cirq.Moment]', system_qubits: Sequence['cirq.Qid']
    ) -> Sequence['cirq.OP_TREE']:
        result = []
        for moment in moments:
            result.append([self.noisy_operation(op) for op in moment])
        return result

    @value.alternative(requires='noisy_moment', implementation=_noisy_moments_impl_moment)
    @value.alternative(requires='noisy_operation', implementation=_noisy_moments_impl_operation)
    def noisy_moments(
        self, moments: 'Iterable[cirq.Moment]', system_qubits: Sequence['cirq.Qid']
    ) -> Sequence['cirq.OP_TREE']:
        """Adds possibly stateful noise to a series of moments.

        Args:
            moments: The moments to add noise to.
            system_qubits: A list of all qubits in the system.

        Returns:
            A sequence of OP_TREEs, with the k'th tree corresponding to the
            noisy operations for the k'th moment.
        """

    def _noisy_moment_impl_moments(
        self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        return self.noisy_moments([moment], system_qubits)

    def _noisy_moment_impl_operation(
        self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        return [self.noisy_operation(op) for op in moment]

    @value.alternative(requires='noisy_moments', implementation=_noisy_moment_impl_moments)
    @value.alternative(requires='noisy_operation', implementation=_noisy_moment_impl_operation)
    def noisy_moment(
        self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        """Adds noise to the operations from a moment.

        Args:
            moment: The moment to add noise to.
            system_qubits: A list of all qubits in the system.

        Returns:
            An OP_TREE corresponding to the noisy operations for the moment.
        """

    def _noisy_operation_impl_moments(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        return self.noisy_moments([ops.Moment([operation])], operation.qubits)

    def _noisy_operation_impl_moment(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        return self.noisy_moment(ops.Moment([operation]), operation.qubits)

    @value.alternative(requires='noisy_moments', implementation=_noisy_operation_impl_moments)
    @value.alternative(requires='noisy_moment', implementation=_noisy_operation_impl_moment)
    def noisy_operation(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        """Adds noise to an individual operation.

        Args:
            operation: The operation to make noisy.

        Returns:
            An OP_TREE corresponding to the noisy operations implementing the
            noisy version of the given operation.
        """


@value.value_equality
class _NoNoiseModel(NoiseModel):
    """A default noise model that adds no noise."""

    def noisy_moments(self, moments: 'Iterable[cirq.Moment]', system_qubits: Sequence['cirq.Qid']):
        return list(moments)

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        return moment

    def noisy_operation(self, operation: 'cirq.Operation'):
        return operation

    def _value_equality_values_(self) -> Any:
        return None

    def __str__(self) -> str:
        return '(no noise)'

    def __repr__(self) -> str:
        return 'cirq.NO_NOISE'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, [])

    def _has_unitary_(self):
        return True

    def _has_mixture_(self):
        return True


@value.value_equality
class ConstantQubitNoiseModel(NoiseModel):
    """Applies noise to each qubit individually at the start of every moment.

    This is the noise model that is wrapped around an operation when that
    operation is given as "the noise to use" for a `NOISE_MODEL_LIKE` parameter.
    """

    def __init__(self, qubit_noise_gate: 'cirq.Gate'):
        if qubit_noise_gate.num_qubits() != 1:
            raise ValueError('noise.num_qubits() != 1')
        self.qubit_noise_gate = qubit_noise_gate

    def _value_equality_values_(self) -> Any:
        return self.qubit_noise_gate

    def __repr__(self) -> str:
        return f'cirq.ConstantQubitNoiseModel({self.qubit_noise_gate!r})'

    def noisy_moment(self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']):
        # Noise should not be appended to previously-added noise.
        if self.is_virtual_moment(moment):
            return moment
        return [
            moment,
            ops.Moment(
                [self.qubit_noise_gate(q).with_tags(ops.VirtualTag()) for q in system_qubits]
            ),
        ]

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['qubit_noise_gate'])

    def _has_unitary_(self):
        return protocols.has_unitary(self.qubit_noise_gate)

    def _has_mixture_(self):
        return protocols.has_mixture(self.qubit_noise_gate)


class GateSubstitutionNoiseModel(NoiseModel):
    def __init__(self, substitution_func: Callable[['cirq.Operation'], 'cirq.Operation']):
        self.substitution_func = substitution_func

    def noisy_moment(
        self, moment: 'cirq.Moment', system_qubits: Sequence['cirq.Qid']
    ) -> 'cirq.OP_TREE':
        return ops.Moment([self.substitution_func(op) for op in moment.operations])


NO_NOISE: 'cirq.NoiseModel' = _NoNoiseModel()
document(
    NO_NOISE,
    """The trivial noise model with no effects.

    This is the noise model used when a `NOISE_MODEL_LIKE` noise parameter is
    set to `None`.
    """,
)

NOISE_MODEL_LIKE = Union[None, 'cirq.NoiseModel', 'cirq.SingleQubitGate']
document(
    NOISE_MODEL_LIKE,  # type: ignore
    """A `cirq.NoiseModel` or a value that can be trivially converted into one.

    `None` is a `NOISE_MODEL_LIKE`. It will be replaced by the `cirq.NO_NOISE`
    noise model.

    A single qubit gate is a `NOISE_MODEL_LIKE`. It will be wrapped inside of a
    `cirq.ConstantQubitNoiseModel`.
    """,
)


def validate_all_measurements(moment: 'cirq.Moment') -> bool:
    """Ensures that the moment is homogenous and returns whether all ops are measurement gates.

    Args:
        moment: the moment to be checked
    Returns:
        bool: True if all operations are measurements, False if none of them are
    Raises:
        ValueError: If a moment is a mixture of measurement and non-measurement gates.
    """
    cases = {protocols.is_measurement(gate) for gate in moment}
    if len(cases) == 2:
        raise ValueError("Moment must be homogeneous: all measurements or all operations.")
    return True in cases
