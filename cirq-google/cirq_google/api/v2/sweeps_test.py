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

from typing import Iterator

import pytest
import sympy

import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2


class UnknownSweep(sweeps.SingleSweep):
    def _tuple(self):  # pragma: no cover
        return self.key, tuple(range(10))

    def __len__(self) -> int:
        return 10

    def _values(self) -> Iterator[float]:
        return iter(range(10))


@pytest.mark.parametrize(
    'sweep',
    [
        cirq.UnitSweep,
        cirq.Linspace('a', 0, 10, 100),
        cirq.Linspace(
            'a',
            0,
            10,
            100,
            metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=2, units='ns'),
        ),
        cirq.Points('b', [1, 1.5, 2, 2.5, 3]),
        cirq.Points(
            'b',
            [1, 1.5, 2, 2.5, 3],
            metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=2, units='GHz'),
        ),
        cirq.Points(
            'b',
            [1, 1.5, 2, 2.5, 3],
            metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=None),
        ),
        cirq.Linspace('a', 0, 1, 5) * cirq.Linspace('b', 0, 1, 5),
        cirq.Points('a', [1, 2, 3]) + cirq.Linspace('b', 0, 1, 3),
        (
            cirq.Linspace('a', 0, 1, 3)
            * (cirq.Linspace('b', 0, 1, 4) + cirq.Linspace('c', 0, 10, 4))
            * (cirq.Linspace('d', 0, 1, 5) + cirq.Linspace('e', 0, 10, 5))
            * (
                cirq.Linspace('f', 0, 1, 6)
                + (cirq.Points('g', [1, 2]) * cirq.Points('h', [-1, 0, 1]))
            )
        ),
    ],
)
def test_sweep_to_proto_roundtrip(sweep):
    msg = v2.sweep_to_proto(sweep)
    deserialized = v2.sweep_from_proto(msg)
    assert deserialized == sweep
    # Check that metadata is the same, if it exists.
    assert getattr(deserialized, 'metadata', None) == getattr(sweep, 'metadata', None)


def test_sweep_to_proto_linspace():
    proto = v2.sweep_to_proto(
        cirq.Linspace(
            'foo', 0, 1, 20, metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=2)
        )
    )
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert proto.HasField('single_sweep')
    assert proto.single_sweep.parameter_key == 'foo'
    assert proto.single_sweep.WhichOneof('sweep') == 'linspace'
    assert proto.single_sweep.linspace.first_point == 0
    assert proto.single_sweep.linspace.last_point == 1
    assert proto.single_sweep.linspace.num_points == 20
    assert proto.single_sweep.parameter.path == ['path', 'to', 'parameter']
    assert proto.single_sweep.parameter.idx == 2
    assert v2.sweep_from_proto(proto).metadata == DeviceParameter(
        path=['path', 'to', 'parameter'], idx=2
    )


def test_list_sweep_bad_expression():
    with pytest.raises(TypeError, match='formula'):
        _ = cirq.ListSweep([cirq.ParamResolver({sympy.Symbol('a') + sympy.Symbol('b'): 4.0})])


def test_symbol_to_string_conversion():
    sweep = cirq.ListSweep([cirq.ParamResolver({sympy.Symbol('a'): 4.0})])
    proto = v2.sweep_to_proto(sweep)
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    expected = v2.run_context_pb2.Sweep()
    expected.sweep_function.function_type = v2.run_context_pb2.SweepFunction.ZIP
    p1 = expected.sweep_function.sweeps.add()
    p1.single_sweep.parameter_key = 'a'
    p1.single_sweep.points.points.extend([4.0])
    assert proto == expected


def test_sweep_to_proto_points():
    proto = v2.sweep_to_proto(cirq.Points('foo', [-1, 0, 1, 1.5]))
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert proto.HasField('single_sweep')
    assert proto.single_sweep.parameter_key == 'foo'
    assert proto.single_sweep.WhichOneof('sweep') == 'points'
    assert list(proto.single_sweep.points.points) == [-1, 0, 1, 1.5]


def test_sweep_to_proto_unit():
    proto = v2.sweep_to_proto(cirq.UnitSweep)
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert not proto.HasField('single_sweep')
    assert not proto.HasField('sweep_function')


def test_sweep_from_proto_unknown_sweep_type():
    with pytest.raises(ValueError, match='cannot convert to v2 Sweep proto'):
        v2.sweep_to_proto(UnknownSweep('foo'))


def test_sweep_from_proto_sweep_function_not_set():
    proto = v2.run_context_pb2.Sweep()
    proto.sweep_function.sweeps.add()
    with pytest.raises(ValueError, match='invalid sweep function type'):
        v2.sweep_from_proto(proto)


def test_sweep_from_proto_single_sweep_type_not_set():
    proto = v2.run_context_pb2.Sweep()
    proto.single_sweep.parameter_key = 'foo'
    with pytest.raises(ValueError, match='single sweep type not set'):
        v2.sweep_from_proto(proto)


def test_sweep_with_list_sweep():
    ls = cirq.study.to_sweep([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    proto = v2.sweep_to_proto(ls)
    expected = v2.run_context_pb2.Sweep()
    expected.sweep_function.function_type = v2.run_context_pb2.SweepFunction.ZIP
    p1 = expected.sweep_function.sweeps.add()
    p1.single_sweep.parameter_key = 'a'
    p1.single_sweep.points.points.extend([1, 3])
    p2 = expected.sweep_function.sweeps.add()
    p2.single_sweep.parameter_key = 'b'
    p2.single_sweep.points.points.extend([2, 4])
    assert proto == expected


def test_sweep_with_flattened_sweep():
    q = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit(
        cirq.PhasedXPowGate(
            exponent=sympy.Symbol('t') / 4 + 0.5,
            phase_exponent=sympy.Symbol('t') / 2 + 0.1,
            global_shift=0.0,
        )(q),
        cirq.measure(q, key='m'),
    )
    param_sweep1 = cirq.Linspace('t', start=0, stop=1, length=20)
    (_, param_sweep2) = cirq.flatten_with_sweep(circuit, param_sweep1)
    assert v2.sweep_to_proto(param_sweep2) is not None


@pytest.mark.parametrize('pass_out', [False, True])
def test_run_context_to_proto(pass_out: bool) -> None:
    msg = v2.run_context_pb2.RunContext() if pass_out else None
    out = v2.run_context_to_proto(None, 10, out=msg)
    if pass_out:
        assert out is msg
    assert len(out.parameter_sweeps) == 1
    assert v2.sweep_from_proto(out.parameter_sweeps[0].sweep) == cirq.UnitSweep
    assert out.parameter_sweeps[0].repetitions == 10

    sweep = cirq.Linspace('a', 0, 1, 21)
    msg = v2.run_context_pb2.RunContext() if pass_out else None
    out = v2.run_context_to_proto(sweep, 100, out=msg)
    if pass_out:
        assert out is msg
    assert len(out.parameter_sweeps) == 1
    assert v2.sweep_from_proto(out.parameter_sweeps[0].sweep) == sweep
    assert out.parameter_sweeps[0].repetitions == 100


@pytest.mark.parametrize('pass_out', [False, True])
def test_batch_run_context_to_proto(pass_out: bool) -> None:
    msg = v2.batch_pb2.BatchRunContext() if pass_out else None
    out = v2.batch_run_context_to_proto([], out=msg)
    if pass_out:
        assert out is msg
    assert len(out.run_contexts) == 0

    msg = v2.batch_pb2.BatchRunContext() if pass_out else None
    out = v2.batch_run_context_to_proto([(None, 10)], out=msg)
    if pass_out:
        assert out is msg
    assert len(out.run_contexts) == 1
    sweep_message = out.run_contexts[0].parameter_sweeps[0]
    assert v2.sweep_from_proto(sweep_message.sweep) == cirq.UnitSweep
    assert sweep_message.repetitions == 10

    sweep = cirq.Linspace('a', 0, 1, 21)
    msg = v2.batch_pb2.BatchRunContext() if pass_out else None
    out = v2.batch_run_context_to_proto([(None, 10), (sweep, 100)], out=msg)
    if pass_out:
        assert out is msg
    assert len(out.run_contexts) == 2
    sweep_message0 = out.run_contexts[0].parameter_sweeps[0]
    assert v2.sweep_from_proto(sweep_message0.sweep) == cirq.UnitSweep
    assert sweep_message0.repetitions == 10
    sweep_message1 = out.run_contexts[1].parameter_sweeps[0]
    assert v2.sweep_from_proto(sweep_message1.sweep) == sweep
    assert sweep_message1.repetitions == 100
