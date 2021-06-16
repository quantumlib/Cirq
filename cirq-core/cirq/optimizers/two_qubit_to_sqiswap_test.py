import itertools
import numpy as np
import pytest

import cirq


def perturb_unitary(u, amount=1e-10):
    kak = cirq.kak_decomposition(u)
    xyz = np.array(kak.interaction_coefficients) + np.random.uniform(-amount, amount, size=3)

    return cirq.unitary(
        cirq.KakDecomposition(
            global_phase=kak.global_phase,
            single_qubit_operations_before=kak.single_qubit_operations_before,
            single_qubit_operations_after=kak.single_qubit_operations_after,
            interaction_coefficients=tuple(xyz),
        )
    )


def random_unitary(seed):
    return cirq.testing.random_unitary(4, random_state=seed)


def random_locals(x, y, z, seed=None):
    rng = np.random.RandomState(seed)
    a0 = cirq.testing.random_unitary(2, random_state=rng)
    a1 = cirq.testing.random_unitary(2, random_state=rng)
    b0 = cirq.testing.random_unitary(2, random_state=rng)
    b1 = cirq.testing.random_unitary(2, random_state=rng)
    return cirq.unitary(
        cirq.KakDecomposition(
            interaction_coefficients=(x, y, z),
            single_qubit_operations_before=(a0, a1),
            single_qubit_operations_after=(b0, b1),
        )
    )


THREE_SQISWAP_UNITARIES = [
    cirq.unitary(cirq.SWAP),
    cirq.unitary(cirq.FSimGate(theta=np.pi / 6, phi=np.pi / 6)),
    *[perturb_unitary(cirq.unitary(cirq.SWAP)) for _ in range(3)],
    *[
        perturb_unitary(cirq.unitary(cirq.FSimGate(theta=np.pi / 6, phi=np.pi / 6)))
        for _ in range(3)
    ],
    random_unitary(97154),
    random_unitary(45375),
    random_unitary(78061),
    random_unitary(61474),
    random_unitary(22897),
]
TWO_SQISWAP_UNITARIES = [
    cirq.unitary(cirq.XX ** 0.25),
    cirq.unitary(cirq.YY ** 0.7),
    cirq.unitary(cirq.ZZ ** 0.15),
    cirq.unitary(cirq.CNOT),
    cirq.unitary(cirq.ISWAP),
    cirq.unitary(cirq.ISWAP ** 0.1),
    *[perturb_unitary(cirq.unitary(cirq.XX ** 0.25)) for _ in range(3)],
    *[perturb_unitary(cirq.unitary(cirq.YY ** 0.7)) for _ in range(3)],
    *[perturb_unitary(cirq.unitary(cirq.ZZ ** 0.15)) for _ in range(3)],
    *[perturb_unitary(cirq.unitary(cirq.CNOT)) for _ in range(3)],
    *[perturb_unitary(cirq.unitary(cirq.ISWAP)) for _ in range(3)],
    random_unitary(17323),
    random_unitary(99415),
    random_unitary(65832),
    random_unitary(45373),
    random_unitary(27123),
]
ONE_SQISWAP_UNITARIES = [
    cirq.unitary(cirq.SQISWAP),
    cirq.unitary(cirq.ISwapPowGate(exponent=-0.5, global_shift=1)),
    random_locals(np.pi / 8, np.pi / 8, 0, seed=80342),
    random_locals(np.pi / 8, np.pi / 8, 0, seed=83551),
    random_locals(np.pi / 8, np.pi / 8, 0, seed=52450),
]
ONE_SQISWAP_UNITARIES += [perturb_unitary(u, 1e-10) for u in ONE_SQISWAP_UNITARIES]
ZERO_UNITARIES = [
    cirq.unitary(cirq.IdentityGate(2)),
    random_locals(0, 0, 0, seed=62933),
    random_locals(0, 0, 0, seed=50590),
]
ZERO_UNITARIES += [perturb_unitary(u, 1e-10) for u in ZERO_UNITARIES]

# Lists of seeds for cirq.testing.random_unitary(4, random_state=seed)
# corresponding to different regions in the Weyl chamber and possible
# eigenvalue crossings in the 3-SQISWAP decomposition
# Key: 1 (fig. 4):  x < y + abs(z)
#      2 (fig. 7a): x > pi/8
#      4 (fig. 7b): z < 0
#      8 (fig. 7c): y+abs(z) < pi/8 if x <= pi/8 else y+abs(z) < pi/4
WEYL_REGION_UNITARIES = {
    1: list(map(random_unitary, [49903, 84819, 84769])),
    3: list(map(random_unitary, [89483, 79607, 96012])),
    5: list(map(random_unitary, [31979, 78758, 47661])),
    7: list(map(random_unitary, [46429, 80635, 49695])),
    8: list(map(random_unitary, [86872, 91499, 47843])),
    9: list(map(random_unitary, [48333, 48333, 95468])),
    10: list(map(random_unitary, [97633, 99062, 44397])),
    11: list(map(random_unitary, [42319, 41651, 11322])),
    12: list(map(random_unitary, [48530, 64117, 31880])),
    13: list(map(random_unitary, [95140, 13781, 60879])),
    14: list(map(random_unitary, [85837, 14090, 73607])),
    15: list(map(random_unitary, [67955, 86488, 24698])),
}
ALL_REGION_UNITARIES = list(itertools.chain(*WEYL_REGION_UNITARIES.values()))


def assert_valid_decomp(
    u_target,
    operations,
    *,
    check_global_phase,
    single_qubit_gate_types=(cirq.MatrixGate,),
    atol=1e-8,
    qubit_order=cirq.QubitOrder.DEFAULT,
):
    # Check expected gates
    saw_global_phase = False
    for op in operations:
        if len(op.qubits) == 0 and isinstance(op, cirq.GlobalPhaseOperation):
            saw_global_phase = True
            assert check_global_phase, 'Global phase operation was output when it should not.'
        elif len(op.qubits) == 1 and isinstance(op.gate, single_qubit_gate_types):
            pass
        elif len(op.qubits) == 2 and op.gate == cirq.SQISWAP:
            pass
        else:
            assert False, f'Disallowed operation was output: {op}'
    if check_global_phase:
        assert saw_global_phase, 'Global phase operation should have been output.'

    # Compare unitaries
    c = cirq.Circuit(operations)
    u_decomp = c.unitary(qubit_order)
    print(c)
    print(u_target.round(4))
    print(u_decomp.round(4))
    assert cirq.equal_up_to_global_phase(
        u_target, u_decomp
    ), 'Invalid decomposition.  Unitaries do not match.'
    if check_global_phase:
        assert np.allclose(u_target, u_decomp, atol=atol, rtol=0), 'Bad global phase'


def assert_specific_sqiswap_count(operations, count):
    actual = sum(len(op.qubits) == 2 for op in operations)
    assert actual == count, f'Incorrect SQISWAP count.  Expected {count} but got {actual}.'


@pytest.mark.parametrize('cnt', [-1, 4, 10])
def test_invalid_required_sqiswap_count(cnt):
    u = TWO_SQISWAP_UNITARIES[0]
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='required_sqiswap_count'):
        cirq.two_qubit_matrix_to_sqiswap_operations(
            q0, q1, u, required_sqiswap_count=cnt, clean_operations=False
        )


@pytest.mark.parametrize('u', ZERO_UNITARIES)
def test_decomp0(u):
    # Decompose unitaries into zero SQISWAP gates
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(
        q0, q1, u, required_sqiswap_count=0, clean_operations=False
    )
    assert_valid_decomp(u, ops, check_global_phase=True)
    assert_specific_sqiswap_count(ops, 0)


@pytest.mark.parametrize(
    'u', ONE_SQISWAP_UNITARIES + TWO_SQISWAP_UNITARIES + THREE_SQISWAP_UNITARIES
)
def test_decomp0_invalid(u):
    # Attempt to decompose other unitaries into zero SQISWAP gates
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 0 SQISWAP gates'):
        cirq.two_qubit_matrix_to_sqiswap_operations(
            q0, q1, u, required_sqiswap_count=0, clean_operations=False
        )


@pytest.mark.parametrize('u', ONE_SQISWAP_UNITARIES)
def test_decomp1(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(
        q0, q1, u, required_sqiswap_count=1, clean_operations=False
    )
    assert_valid_decomp(u, ops, check_global_phase=True)
    assert_specific_sqiswap_count(ops, 1)


@pytest.mark.parametrize('u', ZERO_UNITARIES + TWO_SQISWAP_UNITARIES + THREE_SQISWAP_UNITARIES)
def test_decomp1_invalid(u):
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 1 SQISWAP gates'):
        cirq.two_qubit_matrix_to_sqiswap_operations(
            q0, q1, u, required_sqiswap_count=1, clean_operations=False
        )


@pytest.mark.parametrize('u', ZERO_UNITARIES + ONE_SQISWAP_UNITARIES + TWO_SQISWAP_UNITARIES)
def test_decomp2(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(
        q0, q1, u, required_sqiswap_count=2, clean_operations=False
    )
    assert_valid_decomp(u, ops, check_global_phase=True)
    assert_specific_sqiswap_count(ops, 2)


@pytest.mark.parametrize('u', THREE_SQISWAP_UNITARIES)
def test_decomp2_invalid(u):
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='cannot be decomposed into exactly 2 SQISWAP gates'):
        cirq.two_qubit_matrix_to_sqiswap_operations(
            q0, q1, u, required_sqiswap_count=2, clean_operations=False
        )


@pytest.mark.parametrize(
    'u', ZERO_UNITARIES + ONE_SQISWAP_UNITARIES + TWO_SQISWAP_UNITARIES + THREE_SQISWAP_UNITARIES
)
def test_decomp3(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(
        q0, q1, u, required_sqiswap_count=3, clean_operations=False
    )
    assert_valid_decomp(u, ops, check_global_phase=True)
    assert_specific_sqiswap_count(ops, 3)


def test_decomp3_invalid():
    # All two-qubit gates can be synthesized with three SQISWAP gates
    u = cirq.unitary(cirq.X ** 0.2)  # Pass an invalid size unitary
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError, match='Input must correspond to a 4x4 unitary matrix'):
        cirq.two_qubit_matrix_to_sqiswap_operations(
            q0, q1, u, required_sqiswap_count=3, clean_operations=False
        )


@pytest.mark.parametrize('u', TWO_SQISWAP_UNITARIES)
def test_qubit_order(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(
        q1, q0, u, required_sqiswap_count=2, clean_operations=False
    )
    assert_valid_decomp(u, ops, check_global_phase=True, qubit_order=(q1, q0))
    assert_specific_sqiswap_count(ops, 2)


@pytest.mark.parametrize('u', ZERO_UNITARIES)
def test_decomp_optimal0(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(q0, q1, u, clean_operations=False)
    assert_valid_decomp(u, ops, check_global_phase=True)
    assert_specific_sqiswap_count(ops, 0)


@pytest.mark.parametrize('u', ONE_SQISWAP_UNITARIES)
def test_decomp_optimal1(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(q0, q1, u, clean_operations=False)
    assert_valid_decomp(u, ops, check_global_phase=True)
    assert_specific_sqiswap_count(ops, 1)


@pytest.mark.parametrize('u', TWO_SQISWAP_UNITARIES)
def test_decomp_optimal2(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(q0, q1, u, clean_operations=False)
    assert_valid_decomp(u, ops, check_global_phase=True)
    assert_specific_sqiswap_count(ops, 2)


@pytest.mark.parametrize('u', THREE_SQISWAP_UNITARIES)
def test_decomp_optimal3(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(q0, q1, u, clean_operations=False)
    assert_valid_decomp(u, ops, check_global_phase=True)
    assert_specific_sqiswap_count(ops, 3)


@pytest.mark.parametrize('u', ALL_REGION_UNITARIES)
def test_all_weyl_regions(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(q0, q1, u, clean_operations=False)
    assert_valid_decomp(u, ops, check_global_phase=True)


@pytest.mark.parametrize('u', THREE_SQISWAP_UNITARIES[:1])
def test_not_clean_operations(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(q0, q1, u, clean_operations=False)
    assert_valid_decomp(u, ops, check_global_phase=True, single_qubit_gate_types=(cirq.MatrixGate,))
    assert_specific_sqiswap_count(ops, 3)


@pytest.mark.parametrize('u', THREE_SQISWAP_UNITARIES)
def test_clean_operations(u):
    q0, q1 = cirq.LineQubit.range(2)
    ops = cirq.two_qubit_matrix_to_sqiswap_operations(q0, q1, u, clean_operations=True)
    assert_valid_decomp(
        u,
        ops,
        check_global_phase=False,
        single_qubit_gate_types=(cirq.PhasedXPowGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate),
    )
    assert_specific_sqiswap_count(ops, 3)
