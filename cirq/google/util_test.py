import pytest

import cirq
from cirq.google import util


@pytest.mark.parametrize('sweep', [
    cirq.UnitSweep,
    cirq.Linspace('a', 0, 10, 100),
    cirq.Points('b', [1, 1.5, 2, 2.5, 3]),
    cirq.Linspace('a', 0, 1, 5) * cirq.Linspace('b', 0, 1, 5),
    cirq.Points('a', [1, 2, 3]) + cirq.Linspace('b', 0, 1, 3),
    ])
def test_sweep_to_proto(sweep):
    msg = util.sweep_to_proto(sweep)
    deserialized = util.sweep_from_proto(msg)
    assert deserialized == sweep
