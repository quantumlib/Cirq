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
import math
from copy import deepcopy
from typing import Iterator

import pytest
import sympy
import tunits

import cirq
from cirq.study import sweeps
from cirq_google.api import v2
from cirq_google.study import DeviceParameter, Metadata


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
            'a',
            [1, 1.5, 2, 2.5, 3],
            metadata=Metadata(
                device_parameters=[DeviceParameter(path=['path', 'to', 'parameter'], idx=2)],
                label="bb",
            ),
        ),
        cirq.Points(
            'a',
            [1],
            metadata=Metadata(
                device_parameters=[
                    DeviceParameter(path=['path', 'to', 'parameter']),
                    DeviceParameter(path=['path', 'to', 'parameter2']),
                ],
                label="bb",
                is_const=True,
            ),
        ),
        cirq.Linspace('a', 0, 10, 100, metadata=Metadata(is_const=True)),
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
        cirq.ZipLongest(cirq.Points('a', [1.0, 2.0, 3.0]), cirq.Points('b', [1.0])),
        # Sweep with constant. Type ignore is because cirq.Points type annotated with floats.
        cirq.Points('a', [None]),  # type: ignore[list-item]
        cirq.Points('a', [None]) * cirq.Points('b', [1, 2, 3]),  # type: ignore[list-item]
        cirq.Points('a', [None]) + cirq.Points('b', [2]),  # type: ignore[list-item]
        cirq.Points('a', [1]),
        cirq.Points('b', [1.0]),
        cirq.Points('c', ["abc"]),  # type: ignore[list-item]
        (
            cirq.Points('a', [1]) * cirq.Points('b', [1.0])
            + cirq.Points('c', ["abc"]) * cirq.Points("d", [1, 2, 3, 4])  # type: ignore[list-item]
        ),
        cirq.Concat(cirq.Points('a', [1.0, 2.0, 3.0]), cirq.Points('a', [4.0])),
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


@pytest.mark.parametrize("val", [None, 1, 1.5, 's'])
def test_build_recover_const(val):
    val2 = v2.sweeps._recover_sweep_const(v2.sweeps._build_sweep_const(val))
    if isinstance(val, float):
        assert math.isclose(val, val2)  # avoid the floating precision issue.
    else:
        assert val2 == val


def test_build_const_unsupported_type():
    with pytest.raises(ValueError, match='Unsupported type for serializing const sweep'):
        v2.sweeps._build_sweep_const((1, 2))


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
    p1.single_sweep.const_value.float_value = 4.0
    assert proto == expected


def test_sweep_to_proto_points():
    proto = v2.sweep_to_proto(cirq.Points('foo', [-1, 0, 1, 1.5]))
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert proto.HasField('single_sweep')
    assert proto.single_sweep.parameter_key == 'foo'
    assert proto.single_sweep.WhichOneof('sweep') == 'points'
    assert list(proto.single_sweep.points.points) == [-1, 0, 1, 1.5]


def test_sweep_to_proto_with_simple_func_succeeds():
    def func(sweep: sweeps.SingleSweep):
        if isinstance(sweep, cirq.Points):
            sweep.points = [point + 3 for point in sweep.points]

        return sweep

    sweep = cirq.Points('foo', [1, 2, 3])
    proto = v2.sweep_to_proto(sweep, sweep_transformer=func)

    assert list(proto.single_sweep.points.points) == [4.0, 5.0, 6.0]


def test_sweep_to_proto_with_func_linspace():
    def func(sweep: sweeps.SingleSweep):
        return cirq.Linspace('foo', 3 * tunits.ns, 6 * tunits.ns, 3)  # type: ignore[arg-type]

    sweep = cirq.Linspace('foo', start=1, stop=3, length=3)
    proto = v2.sweep_to_proto(sweep, sweep_transformer=func)

    assert proto.single_sweep.linspace.first_point == 3.0
    assert proto.single_sweep.linspace.last_point == 6.0
    assert tunits.Value.from_proto(proto.single_sweep.linspace.unit) == tunits.ns


def test_sweep_to_proto_with_func_const_value():
    def func(sweep: sweeps.SingleSweep):
        if isinstance(sweep, cirq.Points):
            sweep.points = [point + 3 for point in sweep.points]

        return sweep

    sweep = cirq.Points('foo', points=[1])
    proto = v2.sweep_to_proto(sweep, sweep_transformer=func)

    assert proto.single_sweep.const_value.int_value == 4


@pytest.mark.parametrize('sweep', [(cirq.Points('foo', [1, 2, 3])), (cirq.Points('foo', [1]))])
def test_sweep_to_proto_with_func_round_trip(sweep):
    def add_tunit_func(sweep: sweeps.SingleSweep):
        if isinstance(sweep, cirq.Points):
            sweep.points = [point * tunits.ns for point in sweep.points]  # type: ignore[misc]

        return sweep

    proto = v2.sweep_to_proto(sweep, sweep_transformer=add_tunit_func)
    recovered = v2.sweep_from_proto(proto)

    assert list(recovered.points)[0] == 1 * tunits.ns


def test_sweep_to_proto_unit():
    proto = v2.sweep_to_proto(cirq.UnitSweep)
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert not proto.HasField('single_sweep')
    assert not proto.HasField('sweep_function')


def test_sweep_to_none_const():
    proto = v2.sweep_to_proto(cirq.Points('foo', [None]))
    assert isinstance(proto, v2.run_context_pb2.Sweep)
    assert proto.HasField('single_sweep')
    assert proto.single_sweep.parameter_key == 'foo'
    assert proto.single_sweep.WhichOneof('sweep') == 'const_value'
    assert proto.single_sweep.const_value.is_none


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


@pytest.mark.parametrize('sweep', [cirq.Points('foo', [1, 2, 3]), cirq.Points('foo', [1])])
def test_sweep_from_proto_with_func_succeeds(sweep):
    def add_tunit_func(sweep: sweeps.SingleSweep):
        if isinstance(sweep, cirq.Points):
            sweep.points = [point * tunits.ns for point in sweep.points]  # type: ignore[misc]

        return sweep

    msg = v2.sweep_to_proto(sweep)
    sweep = v2.sweep_from_proto(msg, sweep_transformer=add_tunit_func)

    assert list(sweep.points)[0] == [1.0 * tunits.ns]


@pytest.mark.parametrize('sweep', [cirq.Points('foo', [1, 2, 3]), cirq.Points('foo', [1])])
def test_sweep_from_proto_with_func_round_trip(sweep):
    def add_tunit_func(sweep: sweeps.SingleSweep):
        if isinstance(sweep, cirq.Points):
            sweep.points = [point * tunits.ns for point in sweep.points]  # type: ignore[misc]

        return sweep

    def strip_tunit_func(sweep: sweeps.SingleSweep):
        if isinstance(sweep, cirq.Points):
            if isinstance(sweep.points[0], tunits.Value):
                sweep.points = [point[point.unit] for point in sweep.points]

        return sweep

    msg = v2.sweep_to_proto(sweep, sweep_transformer=add_tunit_func)
    sweep = v2.sweep_from_proto(msg, sweep_transformer=strip_tunit_func)

    assert list(sweep.points)[0] == 1.0


@pytest.mark.parametrize(
    'sweep',
    [
        cirq.Concat(cirq.Points('a', [1, 2, 3]), cirq.Points('a', [4])),
        cirq.Points('a', [1, 2, 3]) * cirq.Points('b', [4, 5, 6]),
        cirq.ZipLongest(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [1])),
        cirq.Zip(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6])),
    ],
)
def test_sweep_to_proto_with_func_on_resursive_sweep_succeeds(sweep):
    def add_tunit_func(sweep: sweeps.SingleSweep):
        if isinstance(sweep, cirq.Points):
            sweep.points = [point * tunits.ns for point in sweep.points]  # type: ignore[misc]

        return sweep

    msg = v2.sweep_to_proto(sweep, sweep_transformer=add_tunit_func)

    assert msg.sweep_function.sweeps[0].single_sweep.points.unit == tunits.ns.to_proto()


@pytest.mark.parametrize(
    'expected_sweep',
    [
        cirq.Concat(cirq.Points('a', [1.0, 2.0, 3.0]), cirq.Points('a', [4.0])),
        cirq.Points('a', [1.0, 2.0, 3.0]) * cirq.Points('b', [4.0, 5.0, 6.0]),
        cirq.ZipLongest(cirq.Points('a', [1.0, 2.0, 3.0]), cirq.Points('b', [1.0])),
        cirq.Zip(cirq.Points('a', [1.0, 2.0, 3.0]), cirq.Points('b', [4.0, 5.0, 6.0])),
        cirq.Points('a', [1, 2, 3])
        + cirq.Points(
            'b',
            [4, 5, 6],
            metadata=DeviceParameter(path=['path', 'to', 'parameter'], idx=2, units='GHz'),
        ),
    ],
)
def test_sweep_from_proto_with_func_on_resursive_sweep_succeeds(expected_sweep):
    def add_tunit_func(sweep_to_transform: sweeps.SingleSweep):
        sweep = deepcopy(sweep_to_transform)
        if isinstance(sweep, cirq.Points):
            sweep.points = [point * tunits.ns for point in sweep.points]  # type: ignore[misc]

        return sweep

    def strip_tunit_func(sweep_to_transform: sweeps.SingleSweep):
        sweep = deepcopy(sweep_to_transform)
        if isinstance(sweep, cirq.Points):
            if isinstance(sweep.points[0], tunits.Value):
                sweep.points = [point[point.unit] for point in sweep.points]

        return sweep

    msg = v2.sweep_to_proto(expected_sweep, sweep_transformer=add_tunit_func)
    round_trip_sweep = v2.sweep_from_proto(msg, strip_tunit_func)

    assert round_trip_sweep == expected_sweep


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


@pytest.mark.parametrize(
    'sweep',
    [
        (cirq.Linspace('tunits_linspace', tunits.ns, 10 * tunits.ns, 15)),  # type: ignore[arg-type]
        (cirq.Points('tunits_points', [tunits.uV, tunits.mV])),  # type: ignore[list-item]
        (cirq.Points('tunits_const', [tunits.MHz])),  # type: ignore[list-item]
    ],
)
def test_tunits_round_trip(sweep):
    msg = v2.sweep_to_proto(sweep)
    recovered = v2.sweep_from_proto(msg)
    assert sweep == recovered
