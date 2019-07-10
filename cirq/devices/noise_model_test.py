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

from typing import Iterable, Sequence, Union

import math
import pytest
import numpy as np

import cirq


def _assert_equivalent_op_tree(x: cirq.OP_TREE, y: cirq.OP_TREE):
    a = list(cirq.flatten_op_tree(x))
    b = list(cirq.flatten_op_tree(y))
    assert a == b


def _assert_equivalent_op_tree_sequence(x: Sequence[cirq.OP_TREE],
                                        y: Sequence[cirq.OP_TREE]):
    assert len(x) == len(y)
    for a, b in zip(x, y):
        _assert_equivalent_op_tree(a, b)


def test_requires_one_override():

    class C(cirq.NoiseModel):
        pass

    with pytest.raises(TypeError, match='abstract'):
        _ = C()


def test_infers_other_methods():
    q = cirq.LineQubit(0)

    class NoiseModelWithNoisyMomentListMethod(cirq.NoiseModel):

        def noisy_moments(self, moments, system_qubits):
            result = []
            for moment in moments:
                if moment.operations:
                    result.append(cirq.X(moment.operations[0].qubits[0]))
                else:
                    result.append([])
            return result

    a = NoiseModelWithNoisyMomentListMethod()
    _assert_equivalent_op_tree(a.noisy_operation(cirq.H(q)), cirq.X(q))
    _assert_equivalent_op_tree(a.noisy_moment(cirq.Moment([cirq.H(q)]), [q]),
                               cirq.X(q))
    _assert_equivalent_op_tree_sequence(
        a.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]),
        [[], cirq.X(q)])

    class NoiseModelWithNoisyMomentMethod(cirq.NoiseModel):

        def noisy_moment(self, moment, system_qubits):
            return cirq.Y.on_each(*moment.qubits)

    b = NoiseModelWithNoisyMomentMethod()
    _assert_equivalent_op_tree(b.noisy_operation(cirq.H(q)), cirq.Y(q))
    _assert_equivalent_op_tree(b.noisy_moment(cirq.Moment([cirq.H(q)]), [q]),
                               cirq.Y(q))
    _assert_equivalent_op_tree_sequence(
        b.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]),
        [[], cirq.Y(q)])

    class NoiseModelWithNoisyOperationMethod(cirq.NoiseModel):

        def noisy_operation(self, operation: 'cirq.Operation'):
            return cirq.Z(operation.qubits[0])

    c = NoiseModelWithNoisyOperationMethod()
    _assert_equivalent_op_tree(c.noisy_operation(cirq.H(q)), cirq.Z(q))
    _assert_equivalent_op_tree(c.noisy_moment(cirq.Moment([cirq.H(q)]), [q]),
                               cirq.Z(q))
    _assert_equivalent_op_tree_sequence(
        c.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]),
        [[], cirq.Z(q)])


def test_no_noise():
    q = cirq.LineQubit(0)
    m = cirq.Moment([cirq.X(q)])
    assert cirq.NO_NOISE.noisy_operation(cirq.X(q)) == cirq.X(q)
    assert cirq.NO_NOISE.noisy_moment(m, [q]) is m
    assert cirq.NO_NOISE.noisy_moments([m, m], [q]) == [m, m]
    assert cirq.NO_NOISE == cirq.NO_NOISE
    assert str(cirq.NO_NOISE) == '(no noise)'
    cirq.testing.assert_equivalent_repr(cirq.NO_NOISE)


def test_constant_qubit_noise():
    a, b, c = cirq.LineQubit.range(3)
    damp = cirq.amplitude_damp(0.5)
    damp_all = cirq.ConstantQubitNoiseModel(damp)
    assert damp_all.noisy_moments(
        [cirq.Moment([cirq.X(a)]), cirq.Moment()],
        [a, b, c]) == [[cirq.X(a), damp(a),
                        damp(b), damp(c)], [damp(a), damp(b),
                                            damp(c)]]

    with pytest.raises(ValueError, match='num_qubits'):
        _ = cirq.ConstantQubitNoiseModel(cirq.CNOT**0.01)


class NonMarkovianTimeSeriesNoise(cirq.NoiseModel):
    """Apply noise to moments sequentially according to a time series.

    In this implementation, apply an DFT to a known noise spectrum ("1/f" noise
    in this case), then approximate evolution by non-commuting operators (noise
    unitary and circuit unitary) via trotterization.
    """
    def __init__(self, epsilon: float, qubit_noise_gate: 'cirq.Gate'):
        """
            Args:
                qubit_noise_gate: The operation representing noise to be applied
                    to each qubit in the moment.
                epsilon: Order of magnitude of noise strength to apply.
        """
        # FIXME: Can't check num qubits on uninstantiated XPowGate
        # if qubit_noise_gate.num_qubits() != 1:
        #     raise ValueError('noise.num_qubits() != 1')
        # FIXME: How to check parametrization on XPowGate
        # if not cirq.protocols.is_parameterized(qubit_noise_gate):
        #     raise ValueError('Noise operation is not parametrized.')
        self.qubit_noise_gate = qubit_noise_gate
        self.epsilon = epsilon

    def noisy_moments(self, moments: 'Iterable[cirq.Moment]',
                      system_qubits: Sequence['cirq.Qid']):
        result = []
        # FT of 1/f spectrum approximates discretized non-Markovian time series.
        frequencies = np.arange(1, len(moments) + 1)
        pulse_train = self.epsilon * np.fft.fft(1 / frequencies)
        for x, moment in zip(pulse_train, moments):
            result.append([self.qubit_noise_gate(x)(q) for q in system_qubits])
        # each sublist in `result` corresponds to a moment parametrized
        # by a single timestep in `pulse_train`
        return [list(moment) for moment in moments] + result


class MarkovianPairwiseCorrelatedNoise(cirq.NoiseModel):
    """Apply noise to a moment following a pairwise distribution over qubits.

    In this implementation, apply the same noise parameter to randomly selected
    pairs of qubits with no interdependence between moments.
    """
    def __init__(self,
                 correlated_pairs: Sequence['cirq.Qid'],
                 epsilon: float,
                 qubit_noise_gate: 'cirq.Gate'):
        """
            Args:
                qubit_noise_gate: The operation representing noise to be applied
                    to each qubit in the moment.
                epsilon: Order of magnitude of noise strength to apply.
        """
        # FIXME: Can't check num qubits on uninstantiated XPowGate
        # if qubit_noise_gate.num_qubits() != 1:
        #     raise ValueError('noise.num_qubits() != 1')
        # FIXME: How to check parametrization on XPowGate
        # if not cirq.protocols.is_parameterized(qubit_noise_gate):
        #     raise ValueError('Noise operation is not parametrized.')
        self.qubit_noise_gate = qubit_noise_gate
        self.epsilon = epsilon

        # Persistent mask describing which qubits share noise at each moment.
        self.correlated_pairs = correlated_pairs

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):

        result = []
        for (i, j) in self.correlated_pairs:
            # This pair will share a single noise parameter.
            shared_x = self.epsilon * np.random.randn()
            for _ in range(2):
                result += [self.qubit_noise_gate(shared_x)(system_qubits[i]),
                           self.qubit_noise_gate(shared_x)(system_qubits[i])]
        return list(moment) + result


class OperationSpecificNoise(cirq.NoiseModel):
    """Apply operation-specific noise.

    In this implementation, specific channels are applied to gates to
    empirically model some hypothetical (fake!) experimental results
    (e.g. process tomography).
    """

    def noisy_operation(self, operation: 'cirq.Operation'):
        # FIXME: better way to get around inheritance to check what gate
        # I'm looking at..?
        if str(operation)[:1] == "X":
            return cirq.depolarize(0.031)(*operation.qubits)
        if str(operation)[:1] == "Y":
            return cirq.depolarize(0.0294)(*operation.qubits)
        if str(operation)[:1] == "Z":
            return cirq.depolarize(0.0299)(*operation.qubits)
        # etc...
        return cirq.I(*operation.qubits) # no modification for other gates...

class CombinationOfNoiseModels(cirq.NoiseModel):
    """Apply a combination of the above noise models.

    In general, noise methods in a cirq.NoiseModel implementation that override
    the abstract methods will be applied in such a way that noise operations are
    applied only to the original operations and Moments in a circuit (and not to
    operations or Moments added by another  overriden abc function). In general,
    the behavior of

        `noisy_circuit = CombinationOfNoiseModels().apply_noise(circuit)`

    differs from the behavior of

        `noisy_circuit = FirstNoiseModel().apply_noise(
            SecondNoiseModel().apply_noise(circuit))`
    """

    def __init__(self,
                 correlated_pairs: Sequence['cirq.Qid'],
                 epsilon: float,
                 qubit_noise_gate: 'cirq.Gate'):

        self._noisy_operation_delegate = OperationSpecificNoise()
        self._noisy_moment_delegate = MarkovianPairwiseCorrelatedNoise(
            correlated_pairs, epsilon, qubit_noise_gate)
        self._noisy_moments_delegate = NonMarkovianTimeSeriesNoise(
            epsilon, qubit_noise_gate)

    def noisy_moments(self, moments: 'Iterable[cirq.Moment]',
                      system_qubits: Sequence['cirq.Qid']):
        return self._noisy_moments_delegate.noisy_moments(moments, system_qubits)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        return self._noisy_moment_delegate.noisy_moment(moment, system_qubits)

    def noisy_operation(self, operation: 'cirq.Operation'):
        return self._noisy_operation_delegate.noisy_operation(operation)


def test_composition_of_noise_models():
    qubits = cirq.LineQubit.range(4)
    epsilon = 0.001
    noise_gate = cirq.Rx
    # This mask randomly pairs qubits for "spatially correlated" noise
    rand_order = np.arange(4, dtype=int)
    np.random.shuffle(rand_order)
    pairs = [rand_order[:2], rand_order[2:4]]


    test_circuit = cirq.Circuit.from_ops([cirq.X(q) for q in qubits]
                                         + [cirq.Y(q) for q in qubits]
                                         + [cirq.Z(q) for q in qubits])
    first_model = NonMarkovianTimeSeriesNoise(epsilon, noise_gate)
    second_model = MarkovianPairwiseCorrelatedNoise(pairs, epsilon, noise_gate)
    third_model = OperationSpecificNoise()

    stacked_noise = test_circuit.copy()
    for noise_model in [first_model, second_model, third_model]:
        stacked_noise = noise_model.apply_noise(stacked_noise)

    composite_model = CombinationOfNoiseModels(pairs, epsilon, noise_gate)
    composite_noise = composite_model.apply_noise(test_circuit)

    assert stacked_noise != composite_noise

if __name__ == "__main__":
    test_composition_of_noise_models()
