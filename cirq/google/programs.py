from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from cirq.api.google.v1 import operations_pb2
from cirq.google import xmon_gates, xmon_gate_ext
from cirq.google.xmon_device import XmonDevice
from cirq.schedules import Schedule, ScheduledOperation
from cirq.value import Timestamp


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


def unpack_results(
        data: bytes,
        repetitions: int,
        key_sizes: Sequence[Tuple[str, int]]
) -> Dict[str, np.ndarray]:
    """Unpack data from a bitstring into individual measurement results.

    Args:
        data: Packed measurement results, in the form <rep0><rep1>...
            where each repetition is <key0_0>..<key0_{size0-1}><key1_0>...
            with bits packed in little-endian order in each byte.
        repetitions: number of repetitions.
        key_sizes: Keys and sizes of the measurements in the data.

    Returns:
        Dict mapping measurement key to a 2D array of boolean results. Each
        array has shape (repetitions, size) with size for that measurement.
    """
    bits_per_rep = sum(size for _, size in key_sizes)
    total_bits = repetitions * bits_per_rep

    byte_arr = np.frombuffer(data, dtype='uint8').reshape((len(data), 1))
    bits = np.unpackbits(byte_arr, axis=1)[:, ::-1].reshape(-1)
    bits = bits[:total_bits].reshape((repetitions, bits_per_rep))

    results = {}
    ofs = 0
    for key, size in key_sizes:
        results[key] = bits[:, ofs:ofs + size]
        ofs += size

    return results
