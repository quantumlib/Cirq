import numpy as np
import pytest

from cirq.testing.random_superposition import random_superposition

@pytest.mark.parametrize('dim',  range(1, 10))
def test_random_superposition(dim):
    state = random_superposition(dim)

    assert dim == len(state)
    assert np.isclose(np.linalg.norm(state), 1.0)
