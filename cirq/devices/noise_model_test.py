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

from typing import Sequence

import numpy as np
import pytest

import cirq
from cirq import ops
from cirq.devices.noise_model import validate_all_measurements


def _assert_equivalent_op_tree(x: cirq.OP_TREE, y: cirq.OP_TREE):
    a = list(cirq.flatten_op_tree(x))
    b = list(cirq.flatten_op_tree(y))
    assert a == b


def _assert_equivalent_op_tree_sequence(x: Sequence[cirq.OP_TREE], y: Sequence[cirq.OP_TREE]):
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
                    result.append(
                        cirq.X(moment.operations[0].qubits[0]).with_tags(ops.VirtualTag())
                    )
                else:
                    result.append([])
            return result

    a = NoiseModelWithNoisyMomentListMethod()
    _assert_equivalent_op_tree(a.noisy_operation(cirq.H(q)), cirq.X(q).with_tags(ops.VirtualTag()))
    _assert_equivalent_op_tree(
        a.noisy_moment(cirq.Moment([cirq.H(q)]), [q]), cirq.X(q).with_tags(ops.VirtualTag())
    )
    _assert_equivalent_op_tree_sequence(
        a.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]),
        [[], cirq.X(q).with_tags(ops.VirtualTag())],
    )

    class NoiseModelWithNoisyMomentMethod(cirq.NoiseModel):
        def noisy_moment(self, moment, system_qubits):
            return [y.with_tags(ops.VirtualTag()) for y in cirq.Y.on_each(*moment.qubits)]

    b = NoiseModelWithNoisyMomentMethod()
    _assert_equivalent_op_tree(b.noisy_operation(cirq.H(q)), cirq.Y(q).with_tags(ops.VirtualTag()))
    _assert_equivalent_op_tree(
        b.noisy_moment(cirq.Moment([cirq.H(q)]), [q]), cirq.Y(q).with_tags(ops.VirtualTag())
    )
    _assert_equivalent_op_tree_sequence(
        b.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]),
        [[], cirq.Y(q).with_tags(ops.VirtualTag())],
    )

    class NoiseModelWithNoisyOperationMethod(cirq.NoiseModel):
        def noisy_operation(self, operation: 'cirq.Operation'):
            return cirq.Z(operation.qubits[0]).with_tags(ops.VirtualTag())

    c = NoiseModelWithNoisyOperationMethod()
    _assert_equivalent_op_tree(c.noisy_operation(cirq.H(q)), cirq.Z(q).with_tags(ops.VirtualTag()))
    _assert_equivalent_op_tree(
        c.noisy_moment(cirq.Moment([cirq.H(q)]), [q]), cirq.Z(q).with_tags(ops.VirtualTag())
    )
    _assert_equivalent_op_tree_sequence(
        c.noisy_moments([cirq.Moment(), cirq.Moment([cirq.H(q)])], [q]),
        [[], cirq.Z(q).with_tags(ops.VirtualTag())],
    )


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
    actual = damp_all.noisy_moments([cirq.Moment([cirq.X(a)]), cirq.Moment()], [a, b, c])
    expected = [
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment(d.with_tags(ops.VirtualTag()) for d in [damp(a), damp(b), damp(c)]),
        ],
        [
            cirq.Moment(),
            cirq.Moment(d.with_tags(ops.VirtualTag()) for d in [damp(a), damp(b), damp(c)]),
        ],
    ]
    assert actual == expected
    cirq.testing.assert_equivalent_repr(damp_all)

    with pytest.raises(ValueError, match='num_qubits'):
        _ = cirq.ConstantQubitNoiseModel(cirq.CNOT ** 0.01)


def test_noise_composition():
    # Verify that noise models can be composed without regard to ordering, as
    # long as the noise operators commute with one another.
    a, b, c = cirq.LineQubit.range(3)
    noise_z = cirq.ConstantQubitNoiseModel(cirq.Z)
    noise_inv_s = cirq.ConstantQubitNoiseModel(cirq.S ** -1)
    merge = cirq.optimizers.merge_single_qubit_gates_into_phased_x_z
    base_moments = [cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(b)]), cirq.Moment([cirq.H(c)])]
    circuit_z = cirq.Circuit(noise_z.noisy_moments(base_moments, [a, b, c]))
    circuit_s = cirq.Circuit(noise_inv_s.noisy_moments(base_moments, [a, b, c]))
    actual_zs = cirq.Circuit(noise_inv_s.noisy_moments(circuit_z.moments, [a, b, c]))
    actual_sz = cirq.Circuit(noise_z.noisy_moments(circuit_s.moments, [a, b, c]))

    expected_circuit = cirq.Circuit(
        cirq.Moment([cirq.X(a)]),
        cirq.Moment([cirq.S(a), cirq.S(b), cirq.S(c)]),
        cirq.Moment([cirq.Y(b)]),
        cirq.Moment([cirq.S(a), cirq.S(b), cirq.S(c)]),
        cirq.Moment([cirq.H(c)]),
        cirq.Moment([cirq.S(a), cirq.S(b), cirq.S(c)]),
    )

    # All of the gates will be the same, just out of order. Merging fixes this.
    merge(actual_zs)
    merge(actual_sz)
    merge(expected_circuit)
    _assert_equivalent_op_tree(actual_zs, actual_sz)
    _assert_equivalent_op_tree(actual_zs, expected_circuit)


def test_constant_qubit_noise_repr():
    cirq.testing.assert_equivalent_repr(cirq.ConstantQubitNoiseModel(cirq.X ** 0.01))


def test_wrap():
    class Forget(cirq.NoiseModel):
        def noisy_operation(self, operation):
            raise NotImplementedError()

    forget = Forget()

    assert cirq.NoiseModel.from_noise_model_like(None) is cirq.NO_NOISE
    assert cirq.NoiseModel.from_noise_model_like(
        cirq.depolarize(0.1)
    ) == cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))
    assert cirq.NoiseModel.from_noise_model_like(cirq.Z ** 0.01) == cirq.ConstantQubitNoiseModel(
        cirq.Z ** 0.01
    )
    assert cirq.NoiseModel.from_noise_model_like(forget) is forget

    with pytest.raises(TypeError, match='Expected a NOISE_MODEL_LIKE'):
        _ = cirq.NoiseModel.from_noise_model_like('test')

    with pytest.raises(ValueError, match='Multi-qubit gate'):
        _ = cirq.NoiseModel.from_noise_model_like(cirq.CZ ** 0.01)


def test_gate_substitution_noise_model():
    def _overrotation(op):
        if isinstance(op.gate, cirq.XPowGate):
            return cirq.XPowGate(exponent=op.gate.exponent + 0.1).on(*op.qubits)
        return op

    noise = cirq.devices.noise_model.GateSubstitutionNoiseModel(_overrotation)

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q0) ** 0.5, cirq.Y(q0))
    circuit2 = cirq.Circuit(cirq.X(q0) ** 0.6, cirq.Y(q0))
    rho1 = cirq.final_density_matrix(circuit, noise=noise)
    rho2 = cirq.final_density_matrix(circuit2)
    np.testing.assert_allclose(rho1, rho2)


def test_moment_is_measurements():
    q = cirq.LineQubit.range(2)
    circ = cirq.Circuit([cirq.X(q[0]), cirq.X(q[1]), cirq.measure(*q, key='z')])
    assert not validate_all_measurements(circ[0])
    assert validate_all_measurements(circ[1])


def test_moment_is_measurements_mixed1():
    q = cirq.LineQubit.range(2)
    circ = cirq.Circuit(
        [
            cirq.X(q[0]),
            cirq.X(q[1]),
            cirq.measure(q[0], key='z'),
            cirq.Z(q[1]),
        ]
    )
    assert not validate_all_measurements(circ[0])
    with pytest.raises(ValueError) as e:
        validate_all_measurements(circ[1])
    assert e.match(".*must be homogeneous: all measurements.*")


def test_moment_is_measurements_mixed2():
    q = cirq.LineQubit.range(2)
    circ = cirq.Circuit(
        [
            cirq.X(q[0]),
            cirq.X(q[1]),
            cirq.Z(q[0]),
            cirq.measure(q[1], key='z'),
        ]
    )
    assert not validate_all_measurements(circ[0])
    with pytest.raises(ValueError) as e:
        validate_all_measurements(circ[1])
    assert e.match(".*must be homogeneous: all measurements.*")
