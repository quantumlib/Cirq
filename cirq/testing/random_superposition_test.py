import numpy as np
import pytest

import cirq.testing

@pytest.mark.parametrize('dim',  range(1, 10))
def test_random_superposition(dim):
    state = cirq.testing.random_superposition(dim)

    assert dim == len(state)
    assert np.isclose(np.linalg.norm(state), 1.0)
