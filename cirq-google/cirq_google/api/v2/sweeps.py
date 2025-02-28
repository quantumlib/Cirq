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

from typing import Any, cast, Callable, Dict, List, Optional

import sympy
import tunits

import cirq
from cirq.study import sweeps
from cirq_google.api.v2 import run_context_pb2
from cirq_google.study.device_parameter import DeviceParameter, Metadata


def _build_sweep_const(value: Any) -> run_context_pb2.ConstValue:
    """Build the sweep const message from a value."""
    if value is None:
        return run_context_pb2.ConstValue(is_none=True)
    elif isinstance(value, float):
        return run_context_pb2.ConstValue(float_value=value)
    elif isinstance(value, int):
        return run_context_pb2.ConstValue(int_value=value)
    elif isinstance(value, str):
        return run_context_pb2.ConstValue(string_value=value)
    elif isinstance(value, tunits.Value):
        return run_context_pb2.ConstValue(with_unit_value=value.to_proto())
    else:
        raise ValueError(
            f"Unsupported type for serializing const sweep: {value=} and {type(value)=}"
        )


def _recover_sweep_const(const_pb: run_context_pb2.ConstValue) -> Any:
    """Recover a const value from the sweep const message."""
    if const_pb.WhichOneof('value') == 'is_none':
        return None
    if const_pb.WhichOneof('value') == 'float_value':
        return const_pb.float_value
    if const_pb.WhichOneof('value') == 'int_value':
        return const_pb.int_value
    if const_pb.WhichOneof('value') == 'string_value':
        return const_pb.string_value
    if const_pb.WhichOneof('value') == 'with_unit_value':
        return tunits.Value.from_proto(const_pb.with_unit_value)


def sweep_to_proto(
    sweep: cirq.Sweep,
    *,
    out: Optional[run_context_pb2.Sweep] = None,
    sweep_transformer: Callable[[sweeps.SingleSweep], sweeps.SingleSweep] = lambda x: x,
) -> run_context_pb2.Sweep:
    """Converts a Sweep to v2 protobuf message.

    Args:
        sweep: The sweep to convert.
        out: Optional message to be populated. If not given, a new message will
            be created.
        sweep_transformer: A function called on Linspace, Points.

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
                factor, out=out.sweep_function.sweeps.add(), sweep_transformer=sweep_transformer
            )
    elif isinstance(sweep, cirq.ZipLongest):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP_LONGEST
        for s in sweep.sweeps:
            sweep_to_proto(
                s, out=out.sweep_function.sweeps.add(), sweep_transformer=sweep_transformer
            )
    elif isinstance(sweep, cirq.Zip):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.ZIP
        for s in sweep.sweeps:
            sweep_to_proto(
                s, out=out.sweep_function.sweeps.add(), sweep_transformer=sweep_transformer
            )
    elif isinstance(sweep, cirq.Concat):
        out.sweep_function.function_type = run_context_pb2.SweepFunction.CONCAT
        for s in sweep.sweeps:
            sweep_to_proto(
                s, out=out.sweep_function.sweeps.add(), sweep_transformer=sweep_transformer
            )
    elif isinstance(sweep, cirq.Linspace) and not isinstance(sweep.key, sympy.Expr):
        sweep = cast(cirq.Linspace, sweep_transformer(sweep))
        out.single_sweep.parameter_key = sweep.key
        if isinstance(sweep.start, tunits.Value):
            unit = sweep.start.unit
            out.single_sweep.linspace.first_point = sweep.start[unit]
            out.single_sweep.linspace.last_point = sweep.stop[unit]
            out.single_sweep.linspace.num_points = sweep.length
            unit.to_proto(out.single_sweep.linspace.unit)
        else:
            out.single_sweep.linspace.first_point = sweep.start
            out.single_sweep.linspace.last_point = sweep.stop
            out.single_sweep.linspace.num_points = sweep.length
        # Encode the metadata if present
        if isinstance(sweep.metadata, Metadata):
            out.single_sweep.metadata.MergeFrom(metadata_to_proto(sweep.metadata))
        else:
            # Use duck-typing to support google-internal Parameter objects
            if sweep.metadata and getattr(sweep.metadata, 'path', None):
                out.single_sweep.parameter.path.extend(sweep.metadata.path)
            if sweep.metadata and getattr(sweep.metadata, 'idx', None):
                out.single_sweep.parameter.idx = sweep.metadata.idx
            if sweep.metadata and getattr(sweep.metadata, 'units', None):
                out.single_sweep.parameter.units = sweep.metadata.units
    elif isinstance(sweep, cirq.Points) and not isinstance(sweep.key, sympy.Expr):
        sweep = cast(cirq.Points, sweep_transformer(sweep))
        out.single_sweep.parameter_key = sweep.key
        if len(sweep.points) == 1:
            out.single_sweep.const_value.MergeFrom(_build_sweep_const(sweep.points[0]))
        else:
            if isinstance(sweep.points[0], tunits.Value):
                unit = sweep.points[0].unit
                out.single_sweep.points.points.extend(p[unit] for p in sweep.points)
                unit.to_proto(out.single_sweep.points.unit)
            else:
                out.single_sweep.points.points.extend(sweep.points)
        # Encode the metadata if present
        if isinstance(sweep.metadata, Metadata):
            out.single_sweep.metadata.MergeFrom(metadata_to_proto(sweep.metadata))
        else:
            # Use duck-typing to support google-internal Parameter objects
            if sweep.metadata and getattr(sweep.metadata, 'path', None):
                out.single_sweep.parameter.path.extend(sweep.metadata.path)
            if sweep.metadata and getattr(sweep.metadata, 'idx', None):
                out.single_sweep.parameter.idx = sweep.metadata.idx
            if sweep.metadata and getattr(sweep.metadata, 'units', None):
                out.single_sweep.parameter.units = sweep.metadata.units
    elif isinstance(sweep, cirq.ListSweep):
        sweep_dict: Dict[str, List[float]] = {}
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
        key = msg.single_sweep.parameter_key
        metadata: DeviceParameter | Metadata | None
        if msg.single_sweep.HasField("parameter"):
            metadata = DeviceParameter(
                path=msg.single_sweep.parameter.path,
                idx=(
                    msg.single_sweep.parameter.idx
                    if msg.single_sweep.parameter.HasField("idx")
                    else None
                ),
                units=(
                    msg.single_sweep.parameter.units
                    if msg.single_sweep.parameter.HasField("units")
                    else None
                ),
            )
        elif msg.single_sweep.HasField("metadata"):
            metadata = metadata_from_proto(msg.single_sweep.metadata)
        else:
            metadata = None

        if msg.single_sweep.WhichOneof('sweep') == 'linspace':
            unit: float | tunits.Value = 1.0
            if msg.single_sweep.linspace.HasField('unit'):
                unit = tunits.Value.from_proto(msg.single_sweep.linspace.unit)
            return sweep_transformer(
                cirq.Linspace(
                    key=key,
                    start=msg.single_sweep.linspace.first_point * unit,  # type: ignore[arg-type]
                    stop=msg.single_sweep.linspace.last_point * unit,  # type: ignore[arg-type]
                    length=msg.single_sweep.linspace.num_points,
                    metadata=metadata,
                )
            )
        if msg.single_sweep.WhichOneof('sweep') == 'points':
            unit = 1.0
            if msg.single_sweep.points.HasField('unit'):
                unit = tunits.Value.from_proto(msg.single_sweep.points.unit)
            return sweep_transformer(
                cirq.Points(
                    key=key,
                    points=[p * unit for p in msg.single_sweep.points.points],
                    metadata=metadata,
                )
            )
        if msg.single_sweep.WhichOneof('sweep') == 'const_value':
            return sweep_transformer(
                cirq.Points(
                    key=key,
                    points=[_recover_sweep_const(msg.single_sweep.const_value)],
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


def run_context_to_proto(
    sweepable: cirq.Sweepable, repetitions: int, *, out: Optional[run_context_pb2.RunContext] = None
) -> run_context_pb2.RunContext:
    """Populates a RunContext protobuf message.

    Args:
        sweepable: The sweepable to include in the run context.
        repetitions: The number of repetitions for the run context.
        out: Optional message to be populated. If not given, a new message will
            be created.

    Returns:
        Populated RunContext protobuf message.
    """
    if out is None:
        out = run_context_pb2.RunContext()
    for sweep in cirq.to_sweeps(sweepable):
        sweep_proto = out.parameter_sweeps.add()
        sweep_proto.repetitions = repetitions
        sweep_to_proto(sweep, out=sweep_proto.sweep)
    return out
