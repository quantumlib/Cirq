# Copyright 2020 The Cirq developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np

from cirq import protocols
from cirq._doc import document

if TYPE_CHECKING:
    import cirq


class _NamedOneQubitState(metaclass=abc.ABCMeta):
    """Abstract class representing a one-qubit state of note."""

    def on(self, qubit) -> 'ProductState':
        """Associates one qubit with this named state.

        The returned object is a ProductState of length 1.
        """
        return ProductState({qubit: self})

    def __call__(self, *args, **kwargs):
        return self.on(*args, **kwargs)

    @abc.abstractmethod
    def state_vector(self):
        """Return a state vector representation of the named state."""

    def projector(self):
        """Return |s⟩⟨s| as a matrix for the named state."""
        vec = self.state_vector()[:, np.newaxis]
        return vec @ vec.conj().T

    def _json_dict_(self):
        # Descendants should be singletons determined solely by the class name.
        # Otherwise, you must override this method.
        return protocols.obj_to_dict_helper(self, [])

    def __eq__(self, other):
        # Descendants should be singletons determined solely by the class name.
        # Otherwise, you must override this method.
        return self.__class__ == other.__class__

    def __hash__(self):
        # Descendants should be singletons determined solely by the class name.
        # Otherwise, you must override this method.
        return hash(self.__class__.__name__)


@dataclass(frozen=True)
class ProductState:
    """A quantum state that is a tensor product of one qubit states.

    For example, the |00⟩ state is `cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1)`.
    The |+⟩ state is a length-1 tensor product state and can be constructed
    with `cirq.KET_PLUS(q0)`.
    """
    states: Dict['cirq.Qid', _NamedOneQubitState]

    def __init__(self, states=None):
        if states is None:
            # coverage: ignore
            states = dict()

        object.__setattr__(self, 'states', states)

    @property
    def qubits(self):
        return sorted(self.states.keys())

    def __mul__(self, other):
        if not isinstance(other, ProductState):
            raise ValueError("Multiplication is only supported "
                             "with other TensorProductStates.")

        dupe_qubits = set(other.states.keys()) & set(self.states.keys())
        if len(dupe_qubits) != 0:
            raise ValueError(
                "You tried to tensor two states, "
                "but both contain factors for these qubits: {}".format(
                    sorted(dupe_qubits)))

        new_states = self.states
        new_states.update(other.states)
        return ProductState(new_states)

    def __str__(self):
        return ' * '.join(f'{st}({q})' for q, st in self.states.items())

    def __repr__(self):
        states_dict_repr = ', '.join(
            f'{repr(key)}: {repr(val)}' for key, val in self.states.items())
        return 'cirq.ProductState({%s})' % states_dict_repr

    def __getitem__(self, qubit):
        """Return the NamedState at the given qubit."""
        return self.states[qubit]

    def __iter__(self):
        yield from self.states.items()

    def __len__(self):
        return len(self.states)

    def __eq__(self, other):
        if not isinstance(other, ProductState):
            return False

        return self.states == other.states

    def __hash__(self):
        return hash(tuple(self.states.items()))

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__,
            'states': list(self.states.items()),
        }

    @classmethod
    def _from_json_dict_(cls, states, **kwargs):
        return cls(states=dict(states))

    def state_vector(self, qubit_order: 'cirq.QubitOrder' = None):
        """The state-vector representation of this state."""
        from cirq import ops
        if qubit_order is None:
            qubit_order = ops.QubitOrder.DEFAULT
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qubits = qubit_order.order_for(self.qubits)

        mat = 1.0 + 0.j
        for qubit in qubits:
            oneq_state = self[qubit]
            state_vector = oneq_state.state_vector()
            mat = np.kron(mat, state_vector)

        return mat

    def projector(self, qubit_order: 'cirq.QubitOrder' = None):
        """The projector associated with this state expressed as a matrix.

        This is |s⟩⟨s| where |s⟩ is this state.
        """
        from cirq import ops
        if qubit_order is None:
            qubit_order = ops.QubitOrder.DEFAULT
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        qubits = qubit_order.order_for(self.qubits)

        mat = 1.0 + 0.j
        for qubit in qubits:
            oneq_state = self[qubit]
            oneq_proj = oneq_state.projector()
            mat = np.kron(mat, oneq_proj)
        return mat


class _KetPlus(_NamedOneQubitState):

    def __str__(self):
        return '+X'

    def __repr__(self):
        return 'cirq.KET_PLUS'

    def state_vector(self):
        return np.array([1, 1]) / np.sqrt(2)

    def stabilized_by(self):
        # Prevent circular import from `value.value_equality`
        from cirq import ops
        return 1, ops.X


class _KetMinus(_NamedOneQubitState):

    def __str__(self):
        return '-X'

    def __repr__(self):
        return 'cirq.KET_MINUS'

    def state_vector(self):
        return np.array([1, -1]) / np.sqrt(2)

    def stabilized_by(self):
        from cirq import ops
        return -1, ops.X


class _KetImag(_NamedOneQubitState):

    def __str__(self):
        return '+Y'

    def __repr__(self):
        return 'cirq.KET_IMAG'

    def state_vector(self):
        return np.array([1, 1j]) / np.sqrt(2)

    def stabilized_by(self):
        from cirq import ops
        return 1, ops.Y


class _KetMinusImag(_NamedOneQubitState):

    def __str__(self):
        return '-Y'

    def __repr__(self):
        return 'cirq.KET_MINUS_IMAG'

    def state_vector(self):
        return np.array([1, -1j]) / np.sqrt(2)

    def stabilized_by(self):
        from cirq import ops
        return -1, ops.Y


class _KetZero(_NamedOneQubitState):

    def __str__(self):
        return '+Z'

    def __repr__(self):
        return 'cirq.KET_ZERO'

    def state_vector(self):
        return np.array([1, 0])

    def stabilized_by(self):
        from cirq import ops
        return 1, ops.Z


class _KetOne(_NamedOneQubitState):

    def __str__(self):
        return '-Z'

    def __repr__(self):
        return 'cirq.KET_ONE'

    def state_vector(self):
        return np.array([0, 1])

    def stabilized_by(self):
        from cirq import ops
        return -1, ops.Z


KET_PLUS = _KetPlus()
document(
    KET_PLUS, """The |+⟩ State
    
    This is the state such that X|+⟩ = +1 |+⟩

    Vector:

        [1, 1] / sqrt(2)
    """)

KET_MINUS = _KetMinus()
document(
    KET_MINUS, """The |-⟩ State
    
    This is the state such that X|-⟩ = -1 |-⟩

    Vector:

        [1, -1] / sqrt(2)
    """)

KET_IMAG = _KetImag()
document(
    KET_IMAG, """The |i⟩ State
    
    This is the state such that Y|i⟩ = +1 |i⟩

    Vector:

        [1, i] / sqrt(2)
    """)

KET_MINUS_IMAG = _KetMinusImag()
document(
    KET_MINUS_IMAG, """The |-i⟩ State

    This is the state such that Y|-i⟩ = -1 |-i⟩

    Vector:

        [1, -i] / sqrt(2)
    """)

KET_ZERO = _KetZero()
document(
    KET_ZERO, """The |0⟩ State

    This is the state such that Z|0⟩ = +1 |0⟩

    Vector:

        [1, 0]
    """)

KET_ONE = _KetOne()
document(
    KET_ONE, """The |1⟩ State

    This is the state such that Z|1⟩ = -1 |1⟩

    Vector:

        [0, 1]
    """)

PAULI_STATES = [
    KET_PLUS,
    KET_MINUS,
    KET_IMAG,
    KET_MINUS_IMAG,
    KET_ZERO,
    KET_ONE,
]
document(PAULI_STATES,
         """All one-qubit states stabalized by the pauli operators.""")
