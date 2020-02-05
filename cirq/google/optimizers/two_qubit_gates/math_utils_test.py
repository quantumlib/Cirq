import numpy as np
import pytest

from cirq import kak_canonicalize_vector, value
from cirq.google.optimizers.two_qubit_gates.math_utils import (
    weyl_chamber_mesh, kak_vector_infidelity, random_qubit_unitary)


def test_weyl_chamber_mesh_spacing_too_small_throws_error():
    with pytest.raises(ValueError, match='may cause system to crash'):
        weyl_chamber_mesh(spacing=5e-4)


def test_kak_vector_infidelity_ignore_equivalent_nontrivial():
    x, y, z = np.pi / 4, 1, 0.5
    kak_0 = kak_canonicalize_vector(x, y, z).interaction_coefficients
    kak_1 = kak_canonicalize_vector(x - 1e-3, y, z).interaction_coefficients

    inf_check_equivalent = kak_vector_infidelity(kak_0, kak_1, False)
    inf_ignore_equivalent = kak_vector_infidelity(kak_0, kak_1, True)

    assert inf_check_equivalent < inf_ignore_equivalent


def test_random_qubit_unitary_shape():
    rng = value.parse_random_state(11)
    actual = random_qubit_unitary((3, 4, 5), True, rng).ravel()
    rng = value.parse_random_state(11)
    expected = random_qubit_unitary((3 * 4 * 5,), True, rng).ravel()
    np.testing.assert_almost_equal(actual, expected)


def test_random_qubit_default():
    rng = value.parse_random_state(11)
    actual = random_qubit_unitary(randomize_global_phase=True, rng=rng).ravel()
    rng = value.parse_random_state(11)
    expected = random_qubit_unitary((1, 1, 1), True, rng=rng).ravel()
    np.testing.assert_almost_equal(actual, expected)
