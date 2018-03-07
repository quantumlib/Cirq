from typing import Iterable

from cirq.api.google.v1 import operations_pb2
from cirq.google import xmon_gates, xmon_gate_ext
from cirq.google.xmon_device import XmonDevice
from cirq.schedules import Schedule, ScheduledOperation
from cirq.time import Timestamp


def schedule_to_proto(schedule: Schedule) -> Iterable[operations_pb2.Operation]:
    """Convert a schedule into protobufs.

    Args:
        schedule: The schedule to convert to protobufs. Must contain only gates
            that can be cast to xmon gates.

    Yields:
        operations_pb2.Operation
    """
    last_time_picos = None  # type: int
    for so in schedule.scheduled_operations:
        gate = xmon_gate_ext.try_cast(so.operation.gate, xmon_gates.XmonGate)
        op = gate.to_proto(*so.operation.qubits)
        time_picos = so.time.raw_picos()
        if last_time_picos is None:
            op.incremental_delay_picoseconds = time_picos
        else:
            op.incremental_delay_picoseconds = time_picos - last_time_picos
        last_time_picos = time_picos
        yield op


def schedule_from_proto(
        device: XmonDevice,
        ops: Iterable[operations_pb2.Operation],
) -> Schedule:
    """Convert protobufs into a Schedule for the given device."""
    scheduled_ops = []
    last_time_picos = 0
    for op in ops:
        time_picos = last_time_picos + op.incremental_delay_picoseconds
        last_time_picos = time_picos
        xmon_op = xmon_gates.XmonGate.from_proto(op)
        scheduled_ops.append(ScheduledOperation.op_at_on(
            operation=xmon_op,
            time=Timestamp(picos=time_picos),
            device=device,
        ))
    return Schedule(device, scheduled_ops)
