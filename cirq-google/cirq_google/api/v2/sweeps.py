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

from __future__ import annotations

import gzip
import numbers
from collections.abc import Callable, Iterable
from typing import Any, cast, TYPE_CHECKING

import sympy
import tunits

import cirq
from cirq_google.api.v2 import run_context_pb2
from cirq_google.study.device_parameter import DeviceParameter, Metadata
from cirq_google.study.finite_random_variable import FiniteRandomVariable

if TYPE_CHECKING:
    from cirq.study import sweeps


def _add_sweep_const(
    sweep: run_context_pb2.SingleSweep, value: Any, use_float64: bool = False
) -> None:
    """Build the sweep const message from a value."""
    if isinstance(value, float):
        # comparing to float is ~5x than testing numbers.Real
        # if modifying the below, also modify the block below for numbers.Real
        if use_float64:
            sweep.const_value.double_value = value
        else:
            # Note: A loss of precision for floating-point numbers may occur here.
            sweep.const_value.float_value = value
    elif isinstance(value, int):
        # comparing to int is ~5x than testing numbers.Integral
        # if modifying the below, also modify the block below for numbers.Integral
        sweep.const_value.int_value = value
    elif value is None:
        sweep.const_value.is_none = True
    elif isinstance(value, str):
        sweep.const_value.string_value = value
    elif isinstance(value, numbers.Integral):
        # more general than isinstance(int) but also slower
        sweep.const_value.int_value = int(value)
    elif isinstance(value, numbers.Real):
        # more general than isinstance(float) but also slower
        if use_float64:
            sweep.const_value.double_value = float(value)  # pragma: no cover
        else:
            # Note: A loss of precision for floating-point numbers may occur here.
            sweep.const_value.float_value = float(value)
    elif isinstance(value, tunits.Value):
        value.to_proto(sweep.const_value.with_unit_value)
    else:
        raise ValueError(
            f"Unsupported type for serializing const sweep: {value=} and {type(value)=}"
        )


def _recover_sweep_const(const_pb: run_context_pb2.ConstValue) -> Any:
    """Recover a const value from the sweep const message."""
    which = const_pb.WhichOneof('value')
    if which == 'is_none':
        return None
    if which == 'double_value':
        return const_pb.double_value
    if which == 'float_value':
        return const_pb.float_value
    if which == 'int_value':
        return const_pb.int_value
    if which == 'string_value':
        return const_pb.string_value
    if which == 'with_unit_value':
        return tunits.Value.from_proto(const_pb.with_unit_value)


def _add_sweep_metadata(sweep: cirq.Sweep, single_sweep: run_context_pb2.SingleSweep) -> None:
    """Encodes the metadata if present and adds Parameter fields if metadata is a Parameter."""
    # Only Linspace, Points, and FiniteRandomVariable sweeps have metadata
    metadata = getattr(sweep, 'metadata', None)
    if isinstance(metadata, Metadata):
        single_sweep.metadata.MergeFrom(metadata_to_proto(metadata))
    elif metadata:
        # Use duck-typing to support google-internal Parameter objects
        if getattr(metadata, 'path', None):
            single_sweep.parameter.path.extend(metadata.path)
        if getattr(metadata, 'idx', None):
            single_sweep.parameter.idx = metadata.idx
        if getattr(metadata, 'units', None):
            single_sweep.parameter.units = metadata.units


def sweep_to_proto(
    sweep: cirq.Sweep,
    *,
    out: run_context_pb2.Sweep | None = None,
    sweep_transformer: Callable[[sweeps.SingleSweep], sweeps.SingleSweep] = lambda x: x,
    use_float64: bool = False,
) -> run_context_pb2.Sweep:
    """Converts a Sweep to v2 protobuf message.

    Args:
        sweep: The sweep to convert.
        out: Optional message to be populated. If not given, a new message will
            be created.
        sweep_transformer: A function called on Linspace, Points.
        use_float64: If true, float64 is used to encode the floating value. If false,
            float32 is used instead. Default: False.

    Returns:
        Populated sweep protobuf message.

    Raises:
        ValueError: If the conversion cannot be completed successfully.
    """
    if out is None:
        out = run_context_pb2.Sweep()
    if sweep is cirq.UnitSweep:
        pass
    elif isinstance(sweep, cirq.Product):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.PRODUCT
        for factor in sweep.factors:
            sweep_to_proto(
                factor,
                out=out.sweep_function.sweeps.add(),
                sweep_transformer=sweep_transformer,
                use_float64=use_float64,
            )
    elif isinstance(sweep, cirq.ZipLongest):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP_LONGEST
        for s in sweep.sweeps:
            sweep_to_proto(
                s,
                out=out.sweep_function.sweeps.add(),
                sweep_transformer=sweep_transformer,
                use_float64=use_float64,
            )
    elif isinstance(sweep, cirq.Zip):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP
        for s in sweep.sweeps:
            sweep_to_proto(
                s,
                out=out.sweep_function.sweeps.add(),
                sweep_transformer=sweep_transformer,
                use_float64=use_float64,
            )
    elif isinstance(sweep, cirq.Concat):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.CONCAT
        for s in sweep.sweeps:
            sweep_to_proto(
                s,
                out=out.sweep_function.sweeps.add(),
                sweep_transformer=sweep_transformer,
                use_float64=use_float64,
            )
    elif isinstance(sweep, cirq.Linspace) and not isinstance(sweep.key, sympy.Expr):
        sweep = cast(cirq.Linspace, sweep_transformer(sweep))
        out.single_sweep.parameter_key = sweep.key
        if isinstance(sweep.start, tunits.Value):
            unit = sweep.start.unit
            if use_float64:
                out.single_sweep.linspace.first_point_double = sweep.start[unit]
                out.single_sweep.linspace.last_point_double = sweep.stop[unit]
            else:
                # Note: A loss of precision for floating-point numbers may occur here.
                out.single_sweep.linspace.first_point = sweep.start[unit]
                out.single_sweep.linspace.last_point = sweep.stop[unit]
            out.single_sweep.linspace.num_points = sweep.length
            unit.to_proto(out.single_sweep.linspace.unit)
        else:
            if use_float64:
                out.single_sweep.linspace.first_point_double = sweep.start
                out.single_sweep.linspace.last_point_double = sweep.stop
            else:
                # Note: A loss of precision for floating-point numbers may occur here.
                out.single_sweep.linspace.first_point = sweep.start
                out.single_sweep.linspace.last_point = sweep.stop

            out.single_sweep.linspace.num_points = sweep.length
        _add_sweep_metadata(sweep, out.single_sweep)
    elif isinstance(sweep, cirq.Points) and not isinstance(sweep.key, sympy.Expr):
        sweep = cast(cirq.Points, sweep_transformer(sweep))
        out.single_sweep.parameter_key = sweep.key
        if len(sweep.points) == 1:
            _add_sweep_const(out.single_sweep, sweep.points[0], use_float64)
        else:
            if isinstance(sweep.points[0], tunits.Value):
                unit = sweep.points[0].unit
                if use_float64:
                    out.single_sweep.points.points_double.extend(p[unit] for p in sweep.points)
                else:
                    # Note: A loss of precision for floating-point numbers may occur here.
                    out.single_sweep.points.points.extend(p[unit] for p in sweep.points)
                unit.to_proto(out.single_sweep.points.unit)
            else:
                if use_float64:
                    out.single_sweep.points.points_double.extend(sweep.points)
                else:
                    # Note: A loss of precision for floating-point numbers may occur here.
                    out.single_sweep.points.points.extend(sweep.points)
        _add_sweep_metadata(sweep, out.single_sweep)
    elif isinstance(sweep, FiniteRandomVariable) and not isinstance(sweep.key, sympy.Expr):
        sweep = cast(FiniteRandomVariable, sweep_transformer(sweep))
        out.single_sweep.parameter_key = sweep.key
        out.single_sweep.random_variable.length = sweep.length
        out.single_sweep.random_variable.seed = sweep.seed
        for random_value, prob in sweep.distribution.items():
            out.single_sweep.random_variable.distribution[str(random_value)] = prob
        _add_sweep_metadata(sweep, out.single_sweep)
    elif isinstance(sweep, cirq.ListSweep):
        sweep_dict: dict[str, list[float]] = {}
        for param_resolver in sweep:
            for key in param_resolver:
                if key not in sweep_dict:
                    sweep_dict[cast(str, key)] = []
                sweep_dict[cast(str, key)].append(cast(float, param_resolver.value_of(key)))
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP
        for key in sweep_dict:
            sweep_to_proto(
                cirq.Points(key, sweep_dict[key]),
                out=out.sweep_function.sweeps.add(),
                sweep_transformer=sweep_transformer,
                use_float64=use_float64,
            )
    else:
        raise ValueError(f'cannot convert to v2 Sweep proto: {sweep}')
    return out


def sweep_from_proto(
    msg: run_context_pb2.Sweep,
    sweep_transformer: Callable[[sweeps.SingleSweep], sweeps.SingleSweep] = lambda x: x,
) -> cirq.Sweep:
    """Creates a Sweep from a v2 protobuf message.

    Args:
        msg: Serialized sweep message.
        sweep_transformer: A function called on Linspace, Point, and ConstValue.

    """
    which = msg.WhichOneof('sweep')
    if which is None:
        return cirq.UnitSweep
    if which == 'sweep_function':
        factors = [sweep_from_proto(m, sweep_transformer) for m in msg.sweep_function.sweeps]
        func_type = msg.sweep_function.function_type
        if func_type == run_context_pb2.SweepFunction.PRODUCT:
            return cirq.Product(*factors)
        if func_type == run_context_pb2.SweepFunction.ZIP:
            return cirq.Zip(*factors)
        if func_type == run_context_pb2.SweepFunction.ZIP_LONGEST:
            return cirq.ZipLongest(*factors)
        if func_type == run_context_pb2.SweepFunction.CONCAT:
            return cirq.Concat(*factors)

        raise ValueError(f'invalid sweep function type: {func_type}')
    if which == 'single_sweep':
        single_sweep = msg.single_sweep
        key = single_sweep.parameter_key
        metadata: DeviceParameter | Metadata | None
        if single_sweep.HasField("parameter"):
            metadata = DeviceParameter(
                path=single_sweep.parameter.path,
                idx=(
                    single_sweep.parameter.idx if single_sweep.parameter.HasField("idx") else None
                ),
                units=(
                    single_sweep.parameter.units
                    if single_sweep.parameter.HasField("units")
                    else None
                ),
            )
        elif single_sweep.HasField("metadata"):
            metadata = metadata_from_proto(single_sweep.metadata)
        else:
            metadata = None

        single_sweep_which = single_sweep.WhichOneof('sweep')
        if single_sweep_which == 'linspace':
            unit: float | tunits.Value = 1.0
            if single_sweep.linspace.HasField('unit'):
                unit = tunits.Value.from_proto(single_sweep.linspace.unit)
            # If float 64 field is presented, we use it first.
            if single_sweep.linspace.first_point_double:
                first_point = single_sweep.linspace.first_point_double
            else:
                first_point = single_sweep.linspace.first_point

            if single_sweep.linspace.last_point_double:
                last_point = single_sweep.linspace.last_point_double
            else:
                last_point = single_sweep.linspace.last_point  # pragma: no cover
            return sweep_transformer(
                cirq.Linspace(
                    key=key,
                    start=first_point * unit,  # type: ignore[arg-type]
                    stop=last_point * unit,  # type: ignore[arg-type]
                    length=single_sweep.linspace.num_points,
                    metadata=metadata,
                )
            )
        if single_sweep_which == 'points':
            # points_double is the double floating number instead of single one.
            # if points_double is presented, we use this value first.
            points_proto = single_sweep.points
            if points_proto.points_double:
                points = points_proto.points_double
            else:
                points = points_proto.points  # pragma: no cover
            if points_proto.HasField('unit'):
                unit = tunits.Value.from_proto(points_proto.unit)
                return sweep_transformer(
                    cirq.Points(key=key, points=[p * unit for p in points], metadata=metadata)
                )
            return sweep_transformer(cirq.Points(key=key, points=points, metadata=metadata))
        if single_sweep_which == 'const_value':
            return sweep_transformer(
                cirq.Points(
                    key=key,
                    points=[_recover_sweep_const(single_sweep.const_value)],
                    metadata=metadata,
                )
            )
        if single_sweep_which == 'random_variable':
            sweep_msg = single_sweep.random_variable
            distribution = {float(key): val for key, val in sweep_msg.distribution.items()}
            return sweep_transformer(
                FiniteRandomVariable(
                    key=key,
                    distribution=distribution,
                    length=sweep_msg.length,
                    seed=sweep_msg.seed,
                    metadata=metadata,
                )
            )

        raise ValueError(f'single sweep type not set: {msg}')

    raise ValueError(f'sweep type not set: {msg}')  # pragma: no cover


def metadata_to_proto(metadata: Metadata) -> run_context_pb2.Metadata:
    """Convert the metadata dataclass to the metadata proto."""
    device_parameters: list[run_context_pb2.DeviceParameter] = []
    if params := getattr(metadata, "device_parameters", None):
        for param in params:
            path = getattr(param, "path", None)
            idx = getattr(param, "idx", None)
            device_parameters.append(run_context_pb2.DeviceParameter(path=path, idx=idx))

    return run_context_pb2.Metadata(
        device_parameters=device_parameters or None,  # If empty set this field as None.
        label=metadata.label,
        is_const=metadata.is_const,
        unit=metadata.unit,
    )


def metadata_from_proto(metadata_pb: run_context_pb2.Metadata) -> Metadata:
    """Convert the metadata proto to the metadata dataclass."""
    device_parameters: list[DeviceParameter] = []
    for param in metadata_pb.device_parameters:
        device_parameters.append(
            DeviceParameter(path=param.path, idx=param.idx if param.HasField("idx") else None)
        )
    return Metadata(
        device_parameters=device_parameters or None,
        label=metadata_pb.label if metadata_pb.HasField("label") else None,
        is_const=metadata_pb.is_const,
        unit=metadata_pb.unit if metadata_pb.HasField("unit") else None,
    )


def sweepable_to_proto(
    sweepable: cirq.Sweepable,
    repetitions: int,
    *,
    out: run_context_pb2.RunContext,
    use_float64: bool = False,
) -> run_context_pb2.RunContext:
    if sweepable is None:
        sweepable = cirq.UnitSweep
    if isinstance(sweepable, cirq.ParamResolver):
        sweepable = sweepable.param_dict or cirq.UnitSweep
    if isinstance(sweepable, cirq.Sweep):
        sweep_proto = out.parameter_sweeps.add()
        sweep_proto.repetitions = repetitions
        sweep_to_proto(sweepable, out=sweep_proto.sweep, use_float64=use_float64)
        return out
    if isinstance(sweepable, dict):
        sweep_proto = out.parameter_sweeps.add()
        sweep_proto.repetitions = repetitions
        zip_proto = sweep_proto.sweep.sweep_function
        zip_proto.function_type = run_context_pb2.SweepFunction.ZIP
        for key, val in sweepable.items():
            single_sweep = zip_proto.sweeps.add().single_sweep
            single_sweep.parameter_key = key
            _add_sweep_const(single_sweep, val, use_float64)
        return out
    if isinstance(sweepable, Iterable):
        for sweepable_element in sweepable:
            sweepable_to_proto(sweepable_element, repetitions, out=out, use_float64=use_float64)
        return out
    raise TypeError(f'Unrecognized sweepable type: {type(sweepable)}.\nsweepable: {sweepable}')


def run_context_to_proto(
    sweepable: cirq.Sweepable,
    repetitions: int,
    *,
    out: run_context_pb2.RunContext | None = None,
    compress_proto: bool = False,
    use_float64: bool = False,
) -> run_context_pb2.RunContext:
    """Populates a RunContext protobuf message.

    Args:
        sweepable: The sweepable to include in the run context.
        repetitions: The number of repetitions for the run context.
        out: Optional message to be populated. If not given, a new message will
            be created.
        compress_proto: If set to `True` the function will gzip the proto and
            store the contents in the bytes field.
        use_float64: If true, float64 is used to encode the floating value. If false,
            float32 is used instead. Default: False.

    Returns:
        Populated RunContext protobuf message.
    """
    if out is None:
        out = run_context_pb2.RunContext()
    if compress_proto:
        uncompressed_wrapper = out
        out = run_context_pb2.RunContext()
    sweepable_to_proto(sweepable, repetitions, out=out, use_float64=use_float64)
    if compress_proto:
        raw_bytes = out.SerializeToString()
        uncompressed_wrapper.compressed_run_context = gzip.compress(raw_bytes)
        return uncompressed_wrapper
    return out
