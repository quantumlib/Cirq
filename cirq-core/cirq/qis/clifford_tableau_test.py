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

"""Tests for clifford tableau."""
import numpy as np
import pytest

import cirq


def _X(table, q):
    table.rs[:] ^= table.zs[:, q]


def _Z(table, q):
    table.rs[:] ^= table.xs[:, q]


def _S(table, q):
    table.rs[:] ^= table.xs[:, q] & table.zs[:, q]
    table.zs[:, q] ^= table.xs[:, q]


def _H(table, q):
    (table.xs[:, q], table.zs[:, q]) = (table.zs[:, q].copy(), table.xs[:, q].copy())
    table.rs[:] ^= table.xs[:, q] & table.zs[:, q]


def _CNOT(table, q1, q2):
    table.rs[:] ^= table.xs[:, q1] & table.zs[:, q2] & (~(table.xs[:, q2] ^ table.zs[:, q1]))
    table.xs[:, q2] ^= table.xs[:, q1]
    table.zs[:, q1] ^= table.zs[:, q2]


@pytest.mark.parametrize('num_qubits', range(1, 4))
def test_tableau_initial_state_string(num_qubits):
    for i in range(2**num_qubits):
        t = cirq.CliffordTableau(initial_state=i, num_qubits=num_qubits)
        splitted_represent_string = str(t).split('\n')
        assert len(splitted_represent_string) == num_qubits
        for n in range(num_qubits):
            sign = '- ' if i >> (num_qubits - n - 1) & 1 else '+ '
            expected_string = sign + 'I ' * n + 'Z ' + 'I ' * (num_qubits - n - 1)
            assert splitted_represent_string[n] == expected_string


def test_stabilizers():
    # Note: the stabilizers are not unique for one state. We just use the one
    # produced by the tableau algorithm.
    # 1. Final state is |1>: Stabalized by -Z.
    t = cirq.CliffordTableau(num_qubits=1, initial_state=1)
    stabilizers = t.stabilizers()
    assert len(stabilizers) == 1
    assert stabilizers[0] == cirq.DensePauliString('Z', coefficient=-1)

    # 2. EPR pair -- Final state is |00> + |11>: Stabalized by XX and ZZ.
    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _CNOT(t, 0, 1)
    stabilizers = t.stabilizers()
    assert len(stabilizers) == 2
    assert stabilizers[0] == cirq.DensePauliString('XX', coefficient=1)
    assert stabilizers[1] == cirq.DensePauliString('ZZ', coefficient=1)

    # 3. Uniform distribution: Stablized by XI and IX.
    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _H(t, 1)
    stabilizers = t.stabilizers()
    assert len(stabilizers) == 2
    assert stabilizers[0] == cirq.DensePauliString('XI', coefficient=1)
    assert stabilizers[1] == cirq.DensePauliString('IX', coefficient=1)


def test_destabilizers():
    # Note: Like stablizers, the destabilizers are not unique for one state, too.
    # We just use the one produced by the tableau algorithm.
    # Under the clifford tableau algorithm, there are several properties that the
    # destablizers have to satisfy:
    #    1. destablizers[i] anti-commutes with stablizers[i]
    #    2. destablizers[i] commutes with destablizers[j] for j!= i
    #    3. destablizers[i] commutes with stablizers[j] for j!= i

    # 1. Final state is |1>: Stabalized by -Z.
    t = cirq.CliffordTableau(num_qubits=1, initial_state=1)
    destabilizers = t.destabilizers()
    assert len(destabilizers) == 1
    assert destabilizers[0] == cirq.DensePauliString('X', coefficient=1)

    # 2. EPR pair -- Final state is |00> + |11>: Stabalized by XX and ZZ.
    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _CNOT(t, 0, 1)
    destabilizers = t.destabilizers()
    assert len(destabilizers) == 2
    assert destabilizers[0] == cirq.DensePauliString('ZI', coefficient=1)
    assert destabilizers[1] == cirq.DensePauliString('IX', coefficient=1)

    # 3. Uniform distribution: Stablized by XI and IX.
    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _H(t, 1)
    destabilizers = t.destabilizers()
    assert len(destabilizers) == 2
    assert destabilizers[0] == cirq.DensePauliString('ZI', coefficient=1)
    assert destabilizers[1] == cirq.DensePauliString('IZ', coefficient=1)


def test_measurement():
    repetitions = 500
    prng = np.random.RandomState(seed=123456)

    # 1. The final state is |0>
    res = []
    for _ in range(repetitions):
        t = cirq.CliffordTableau(num_qubits=1)
        res.append(t._measure(q=0, prng=prng))
    assert all(res) == 0

    # 2. The final state is |1>
    res = []
    for _ in range(repetitions):
        t = cirq.CliffordTableau(num_qubits=1, initial_state=1)
        res.append(t._measure(q=0, prng=prng))
    assert all(res) == 1

    # 3. EPR pair -- The final state is |00> + |11>
    res = []
    for _ in range(repetitions):
        t = cirq.CliffordTableau(num_qubits=2)
        _H(t, 0)
        _CNOT(t, 0, 1)
        res.append(2 * t._measure(q=0, prng=prng) + t._measure(q=1, prng=prng))
    assert set(res) == set([0, 3])
    assert sum(np.asarray(res) == 0) >= (repetitions / 2 * 0.9)
    assert sum(np.asarray(res) == 3) >= (repetitions / 2 * 0.9)

    # 4. Uniform distribution -- The final state is |00> + |01> + |10> + |11>
    res = []
    for _ in range(repetitions):
        t = cirq.CliffordTableau(num_qubits=2)
        _H(t, 0)
        _H(t, 1)
        res.append(2 * t._measure(q=0, prng=prng) + t._measure(q=1, prng=prng))
    assert set(res) == set([0, 1, 2, 3])
    assert sum(np.asarray(res) == 0) >= (repetitions / 4 * 0.9)
    assert sum(np.asarray(res) == 1) >= (repetitions / 4 * 0.9)
    assert sum(np.asarray(res) == 2) >= (repetitions / 4 * 0.9)
    assert sum(np.asarray(res) == 3) >= (repetitions / 4 * 0.9)

    # 5. To cover usage of YY case. The final state is:
    #      0.5|00⟩ + 0.5j|01⟩ + 0.5j|10⟩ - 0.5|11⟩
    res = []
    for _ in range(repetitions):
        t = cirq.CliffordTableau(num_qubits=2)
        _H(t, 0)
        _H(t, 1)  # [ZI, IZ, XI, IX]
        _CNOT(t, 0, 1)  # [ZI, ZZ, XX, IX]
        _S(t, 0)  # [ZI, ZZ, YX, IX]
        _S(t, 1)  # [ZI, ZZ, YY, IY]
        res.append(2 * t._measure(q=0, prng=prng) + t._measure(q=1, prng=prng))
    assert set(res) == set([0, 1, 2, 3])
    assert sum(np.asarray(res) == 0) >= (repetitions / 4 * 0.9)
    assert sum(np.asarray(res) == 1) >= (repetitions / 4 * 0.9)
    assert sum(np.asarray(res) == 2) >= (repetitions / 4 * 0.9)
    assert sum(np.asarray(res) == 3) >= (repetitions / 4 * 0.9)


def test_validate_tableau():
    num_qubits = 4
    for i in range(2**num_qubits):
        t = cirq.CliffordTableau(initial_state=i, num_qubits=num_qubits)
        assert t._validate()

    t = cirq.CliffordTableau(num_qubits=2)
    assert t._validate()
    _H(t, 0)
    assert t._validate()
    _X(t, 0)
    assert t._validate()
    _Z(t, 1)
    assert t._validate()
    _CNOT(t, 0, 1)
    assert t._validate()
    _CNOT(t, 1, 0)
    assert t._validate()

    t.xs = np.zeros((4, 2))
    assert not t._validate()


def test_rowsum():
    # Note: rowsum should not apply on two rows that anti-commute each other.
    t = cirq.CliffordTableau(num_qubits=2)
    # XI * IX ==> XX
    t._rowsum(0, 1)
    assert t.destabilizers()[0] == cirq.DensePauliString('XX', coefficient=1)

    # IX * ZI ==> ZX
    t._rowsum(1, 2)
    assert t.destabilizers()[1] == cirq.DensePauliString('ZX', coefficient=1)

    # ZI * IZ ==> ZZ
    t._rowsum(2, 3)
    assert t.stabilizers()[0] == cirq.DensePauliString('ZZ', coefficient=1)

    t = cirq.CliffordTableau(num_qubits=2)
    _S(t, 0)  # Table now are [YI, IX, ZI, IZ]
    _CNOT(t, 0, 1)  # Table now are [YX, IX, ZI, ZZ]

    # YX * ZZ ==> XY
    t._rowsum(0, 3)
    assert t.destabilizers()[0] == cirq.DensePauliString('XY', coefficient=1)

    # ZZ * XY ==> YX
    t._rowsum(3, 0)
    assert t.stabilizers()[1] == cirq.DensePauliString('YX', coefficient=1)


def test_json_dict():
    t = cirq.CliffordTableau._from_json_dict_(n=1, rs=[0, 0], xs=[[1], [0]], zs=[[0], [1]])
    assert t.destabilizers()[0] == cirq.DensePauliString('X', coefficient=1)
    assert t.stabilizers()[0] == cirq.DensePauliString('Z', coefficient=1)
    json_dict = t._json_dict_()
    except_json_dict = {
        'n': 1,
        'rs': [False, False],
        'xs': [[True], [False]],
        'zs': [[False], [True]],
    }
    assert list(json_dict.keys()) == list(except_json_dict.keys())
    for k, v in except_json_dict.items():
        assert k in json_dict
        if isinstance(v, list):
            assert all(json_dict[k] == v)
        else:
            assert json_dict[k] == v


def test_str():
    t = cirq.CliffordTableau(num_qubits=2)
    splitted_represent_string = str(t).split('\n')
    assert len(splitted_represent_string) == 2
    assert splitted_represent_string[0] == '+ Z I '
    assert splitted_represent_string[1] == '+ I Z '

    _H(t, 0)
    _H(t, 1)
    splitted_represent_string = str(t).split('\n')
    assert len(splitted_represent_string) == 2
    assert splitted_represent_string[0] == '+ X I '
    assert splitted_represent_string[1] == '+ I X '

    _S(t, 0)
    _S(t, 1)
    splitted_represent_string = str(t).split('\n')
    assert len(splitted_represent_string) == 2
    assert splitted_represent_string[0] == '+ Y I '
    assert splitted_represent_string[1] == '+ I Y '


def test_repr():
    t = cirq.CliffordTableau(num_qubits=1)
    assert repr(t) == "stabilizers: [cirq.DensePauliString('Z', coefficient=(1+0j))]"


def test_str_full():
    t = cirq.CliffordTableau(num_qubits=1)
    expected_str = r"""stable | destable
-------+----------
+ Z0   | + X0
"""
    assert t._str_full_() == expected_str

    t = cirq.CliffordTableau(num_qubits=1)
    _S(t, 0)
    expected_str = r"""stable | destable
-------+----------
+ Z0   | + Y0
"""
    assert t._str_full_() == expected_str

    t = cirq.CliffordTableau(num_qubits=2)
    expected_str = r"""stable | destable
-------+----------
+ Z0   | + X0  
+   Z1 | +   X1
"""
    assert t._str_full_() == expected_str


def test_copy():
    t = cirq.CliffordTableau(num_qubits=3, initial_state=3)
    new_t = t.copy()

    assert isinstance(new_t, cirq.CliffordTableau)
    assert t is not new_t
    assert t.rs is not new_t.rs
    assert t.xs is not new_t.xs
    assert t.zs is not new_t.zs
    np.testing.assert_array_equal(t.rs, new_t.rs)
    np.testing.assert_array_equal(t.xs, new_t.xs)
    np.testing.assert_array_equal(t.zs, new_t.zs)

    assert t == t.copy() == t.__copy__()


def _three_identical_table(num_qubits):
    t1 = cirq.CliffordTableau(num_qubits)
    t2 = cirq.CliffordTableau(num_qubits)
    t3 = cirq.CliffordTableau(num_qubits)
    return t1, t2, t3


def test_tableau_then():

    t1, t2, expected_t = _three_identical_table(1)
    assert expected_t == t1.then(t2)

    t1, t2, expected_t = _three_identical_table(1)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_H(t, 0) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)

    t1, t2, expected_t = _three_identical_table(1)
    _ = [_X(t, 0) for t in (t1, expected_t)]
    _ = [_S(t, 0) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    assert expected_t != t2.then(t1)

    t1, t2, expected_t = _three_identical_table(1)
    _ = [_X(t, 0) for t in (t1, expected_t)]
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_Z(t, 0) for t in (t1, expected_t)]
    _ = [_S(t, 0) for t in (t2, expected_t)]
    _ = [_H(t, 0) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    assert expected_t != t2.then(t1)

    t1, t2, expected_t = _three_identical_table(2)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_H(t, 1) for t in (t1, expected_t)]
    _ = [_H(t, 0) for t in (t2, expected_t)]
    _ = [_H(t, 1) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)

    t1, t2, expected_t = _three_identical_table(2)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_CNOT(t, 0, 1) for t in (t1, expected_t)]
    _ = [_S(t, 0) for t in (t2, expected_t)]
    _ = [_X(t, 1) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    assert expected_t != t2.then(t1)

    t1, t2, expected_t = _three_identical_table(2)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_CNOT(t, 0, 1) for t in (t1, expected_t)]
    _ = [_S(t, 1) for t in (t2, expected_t)]
    _ = [_CNOT(t, 1, 0) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)
    assert expected_t != t2.then(t1)

    def random_circuit(num_ops, num_qubits, seed=12345):
        prng = np.random.RandomState(seed)
        candidate_op = [_H, _S, _X, _Z]
        if num_qubits > 1:
            candidate_op = [_H, _S, _X, _Z, _CNOT]

        seq_op = []
        for _ in range(num_ops):
            op = prng.randint(len(candidate_op))
            if op != 4:
                args = (prng.randint(num_qubits),)
            else:
                args = prng.choice(num_qubits, 2, replace=False)
            seq_op.append((candidate_op[op], args))
        return seq_op

    # Do small random circuits test 100 times.
    for seed in range(100):
        t1, t2, expected_t = _three_identical_table(8)
        seq_op = random_circuit(num_ops=20, num_qubits=8, seed=seed)
        for i, (op, args) in enumerate(seq_op):
            if i < 7:
                _ = [op(t, *args) for t in (t1, expected_t)]
            else:
                _ = [op(t, *args) for t in (t2, expected_t)]
        assert expected_t == t1.then(t2)

    # Since merging Clifford Tableau operation is O(n^3),
    # running 100 qubits case is still fast.
    t1, t2, expected_t = _three_identical_table(100)
    seq_op = random_circuit(num_ops=1000, num_qubits=100)
    for i, (op, args) in enumerate(seq_op):
        if i < 350:
            _ = [op(t, *args) for t in (t1, expected_t)]
        else:
            _ = [op(t, *args) for t in (t2, expected_t)]
    assert expected_t == t1.then(t2)


def test_tableau_matmul():
    t1, t2, expected_t = _three_identical_table(1)
    _ = [_H(t, 0) for t in (t1, expected_t)]
    _ = [_H(t, 0) for t in (t2, expected_t)]
    assert expected_t == t2 @ t1

    t1, t2, expected_t = _three_identical_table(1)
    _ = [_X(t, 0) for t in (t1, expected_t)]
    _ = [_S(t, 0) for t in (t2, expected_t)]
    assert expected_t == t2 @ t1
    assert expected_t != t1 @ t2

    with pytest.raises(TypeError):
        # pylint: disable=pointless-statement
        t1 @ 21
        # pylint: enable=pointless-statement


def test_tableau_then_with_bad_input():
    t1 = cirq.CliffordTableau(1)
    t2 = cirq.CliffordTableau(2)
    with pytest.raises(ValueError, match="Mismatched number of qubits of two tableaux: 1 vs 2."):
        t1.then(t2)

    with pytest.raises(TypeError):
        t1.then(cirq.X)


def test_inverse():
    t = cirq.CliffordTableau(num_qubits=1)
    assert t.inverse() == t

    t = cirq.CliffordTableau(num_qubits=1)
    _X(t, 0)
    _S(t, 0)
    expected_t = cirq.CliffordTableau(num_qubits=1)
    _S(expected_t, 0)  # the inverse of S gate is S*S*S.
    _S(expected_t, 0)
    _S(expected_t, 0)
    _X(expected_t, 0)
    assert t.inverse() == expected_t
    assert t.then(t.inverse()) == cirq.CliffordTableau(num_qubits=1)
    assert t.inverse().then(t) == cirq.CliffordTableau(num_qubits=1)

    t = cirq.CliffordTableau(num_qubits=2)
    _H(t, 0)
    _H(t, 1)
    # Because the ops are the same in either forward or backward way,
    # t is self-inverse operator.
    assert t.inverse() == t
    assert t.then(t.inverse()) == cirq.CliffordTableau(num_qubits=2)
    assert t.inverse().then(t) == cirq.CliffordTableau(num_qubits=2)

    t = cirq.CliffordTableau(num_qubits=2)
    _X(t, 0)
    _CNOT(t, 0, 1)
    expected_t = cirq.CliffordTableau(num_qubits=2)
    _CNOT(t, 0, 1)
    _X(t, 0)
    assert t.inverse() == expected_t
    assert t.then(t.inverse()) == cirq.CliffordTableau(num_qubits=2)
    assert t.inverse().then(t) == cirq.CliffordTableau(num_qubits=2)

    def random_circuit_and_its_inverse(num_ops, num_qubits, seed=12345):
        prng = np.random.RandomState(seed)
        candidate_op = [_H, _S, _X, _Z]
        if num_qubits > 1:
            candidate_op = [_H, _S, _X, _Z, _CNOT]

        seq_op = []
        inv_seq_ops = []
        for _ in range(num_ops):
            op = prng.randint(len(candidate_op))
            if op != 4:
                args = (prng.randint(num_qubits),)
            else:
                args = prng.choice(num_qubits, 2, replace=False)
            seq_op.append((candidate_op[op], args))
            if op == 1:  # S gate
                inv_seq_ops.extend([(_S, args), (_S, args), (_S, args)])
            else:
                inv_seq_ops.append((candidate_op[op], args))
        return seq_op, inv_seq_ops[::-1]

    # Do small random circuits test 100 times.
    for seed in range(100):
        t, expected_t, _ = _three_identical_table(7)
        seq_op, inv_seq_ops = random_circuit_and_its_inverse(num_ops=50, num_qubits=7, seed=seed)
        for op, args in seq_op:
            op(t, *args)
        for op, args in inv_seq_ops:
            op(expected_t, *args)
    assert t.inverse() == expected_t
    assert t.then(t.inverse()) == cirq.CliffordTableau(num_qubits=7)
    assert t.inverse().then(t) == cirq.CliffordTableau(num_qubits=7)

    # Since inverse Clifford Tableau operation is O(n^3) (same order of composing two tableaux),
    # running 100 qubits case is still fast.
    t, expected_t, _ = _three_identical_table(100)
    seq_op, inv_seq_ops = random_circuit_and_its_inverse(num_ops=1000, num_qubits=100)
    for op, args in seq_op:
        op(t, *args)
    for op, args in inv_seq_ops:
        op(expected_t, *args)
    assert t.inverse() == expected_t
    assert t.then(t.inverse()) == cirq.CliffordTableau(num_qubits=100)
    assert t.inverse().then(t) == cirq.CliffordTableau(num_qubits=100)
