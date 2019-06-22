import pytest

import cirq
from cirq.api.google import v2
from cirq.google import util


@pytest.mark.parametrize('sweep', [
    cirq.UnitSweep,
    cirq.Linspace('a', 0, 10, 100),
    cirq.Points('b', [1, 1.5, 2, 2.5, 3]),
    cirq.Linspace('a', 0, 1, 5) * cirq.Linspace('b', 0, 1, 5),
    cirq.Points('a', [1, 2, 3]) + cirq.Linspace('b', 0, 1, 3),
    (cirq.Linspace('a', 0, 1, 3) *
     (cirq.Linspace('b', 0, 1, 4) + cirq.Linspace('c', 0, 10, 4)) *
     (cirq.Linspace('d', 0, 1, 5) + cirq.Linspace('e', 0, 10, 5)) *
     (cirq.Linspace('f', 0, 1, 6) +
      (cirq.Points('g', [1, 2]) * cirq.Points('h', [-1, 0, 1])))),
])
def test_sweep_to_proto_roundtrip(sweep):
    msg = util.sweep_to_proto(sweep)
    deserialized = util.sweep_from_proto(msg)
    assert deserialized == sweep


def test_sweep_to_proto_linspace():
    proto = util.sweep_to_proto(cirq.Linspace('foo', 0, 1, 20))
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert proto.HasField('single_sweep')
    assert proto.single_sweep.parameter_key == 'foo'
    assert proto.single_sweep.WhichOneof('sweep') == 'linspace'
    assert proto.single_sweep.linspace.first_point == 0
    assert proto.single_sweep.linspace.last_point == 1
    assert proto.single_sweep.linspace.num_points == 20


def test_sweep_to_proto_points():
    proto = util.sweep_to_proto(cirq.Points('foo', [-1, 0, 1, 1.5]))
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert proto.HasField('single_sweep')
    assert proto.single_sweep.parameter_key == 'foo'
    assert proto.single_sweep.WhichOneof('sweep') == 'points'
    assert list(proto.single_sweep.points.points) == [-1, 0, 1, 1.5]


def test_sweep_to_proto_unit():
    proto = util.sweep_to_proto(cirq.UnitSweep)
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert not proto.HasField('single_sweep')
    assert not proto.HasField('sweep_function')
