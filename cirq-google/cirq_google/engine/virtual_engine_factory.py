# Copyright 2021 The Cirq Developers
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

"""Functions to instantiate SimulatedLocalEngines to simulate various Google Devices."""
from typing import cast, Iterable, Optional
import pathlib
import time

import google.protobuf.text_format as text_format
import cirq
from cirq_google.api import v2
from cirq_google.engine import (
    calibration,
    engine_validator,
    simulated_local_engine,
    simulated_local_processor,
)
from cirq_google.devices import serializable_device
from cirq_google.serialization.gate_sets import FSIM_GATESET
from cirq_google.serialization import serializable_gate_set

METRICS_1Q = [
    'single_qubit_p00_error',
    'single_qubit_p11_error',
    'single_qubit_readout_separation_error',
    'parallel_p00_error',
    'parallel_p11_error',
    'single_qubit_rb_average_error_per_gate',
    'single_qubit_rb_incoherent_error_per_gate',
    'single_qubit_rb_pauli_error_per_gate',
]

METRICS_2Q = [
    'two_qubit_sycamore_gate_xeb_average_error_per_cycle',
    'two_qubit_sycamore_gate_xeb_pauli_error_per_cycle',
    'two_qubit_sycamore_gate_xeb_incoherent_error_per_cycle',
    'two_qubit_sqrt_iswap_gate_xeb_average_error_per_cycle',
    'two_qubit_sqrt_iswap_gate_xeb_pauli_error_per_cycle',
    'two_qubit_sqrt_iswap_gate_xeb_incoherent_error_per_cycle',
    'two_qubit_parallel_sycamore_gate_xeb_average_error_per_cycle',
    'two_qubit_parallel_sycamore_gate_xeb_pauli_error_per_cycle',
    'two_qubit_parallel_sycamore_gate_xeb_incoherent_error_per_cycle',
    'two_qubit_parallel_sqrt_iswap_gate_xeb_average_error_per_cycle',
    'two_qubit_parallel_sqrt_iswap_gate_xeb_pauli_error_per_cycle',
    'two_qubit_parallel_sqrt_iswap_gate_xeb_incoherent_error_per_cycle',
]
# Technically, T1 for a noiseless simulation should be infinite,
# but we will set it to a very high value.
PERFECT_T1_VALUE = 1_000_000
T1_METRIC_NAME = 'single_qubit_idle_t1_micros'


def _create_perfect_calibration(device: cirq.Device):
    all_metrics: calibration.ALL_METRICS = {}
    qubit_set = device.qubit_set()
    if qubit_set is None:
        raise ValueError('Devices for noiseless Virtual Engine must have qubits')
    qubits = [cast(cirq.GridQubit, q) for q in qubit_set]
    for name in METRICS_1Q:
        all_metrics[name] = {(q,): [0.0] for q in qubits}
    for name in METRICS_2Q:
        metric_dict: calibration.METRIC_DICT = {}
        pairs = device.qid_pairs()
        if pairs is not None:
            for pair in pairs:
                qubits = [cast(cirq.GridQubit, q) for q in pair.qids]
                metric_dict[(qubits[0], qubits[1])] = [0.0]
        all_metrics[name] = metric_dict
    all_metrics[T1_METRIC_NAME] = {(q,): [PERFECT_T1_VALUE] for q in qubits}
    snapshot = v2.metrics_pb2.MetricsSnapshot()
    snapshot.timestamp_ms = int(time.time() * 1000)
    return calibration.Calibration(calibration=snapshot, metrics=all_metrics)


def create_noiseless_virtual_engine_from_device(processor_id: str, device: cirq.Device):
    """Creates an Engine object that is backed by a noiseless simulator.

    Creates a noiseless engine object based on the cirq simulator,
    a default validator, and a provided device.

    Args:
         processor_id: name of the processor to simulate.  This is an arbitrary
             string identifier and does not have to match the processor's name
             in QCS.
         device: A `cirq.Device` to validate circuits against.
    """
    calibration = _create_perfect_calibration(device)
    processor = simulated_local_processor.SimulatedLocalProcessor(
        processor_id=processor_id,
        device=device,
        validator=engine_validator.create_engine_validator(),
        gate_set_validator=engine_validator.create_gate_set_validator(),
        calibrations={calibration.timestamp // 1000: calibration},
    )
    return simulated_local_engine.SimulatedLocalEngine([processor])


def create_noiseless_virtual_engine_from_proto(
    processor_id: str,
    device_specification: v2.device_pb2.DeviceSpecification,
    gate_sets: Optional[Iterable[serializable_gate_set.SerializableGateSet]] = None,
):
    """Creates a noiseless virtual engine object from a device specification proto.a

    The device specification protocol buffer specifies qubits and gates on the device
    and can be retrieved from a stored "proto.txt" file or from the QCS API.

    Args:
         processor_id: name of the processor to simulate.  This is an arbitrary
             string identifier and does not have to match the processor's name
             in QCS.
         device_specification:  `v2.device_pb2.DeviceSpecification` proto to create
             a validating device from.
         gate_sets: Iterable of serializers to use in the processor.  Defaults
             to the FSIM_GATESET.
    """
    if gate_sets is None:
        gate_sets = [FSIM_GATESET]
    device = serializable_device.SerializableDevice.from_proto(device_specification, gate_sets)
    return create_noiseless_virtual_engine_from_device(processor_id, device)


def create_noiseless_virtual_engine_from_template(
    processor_id: str,
    template_name: str,
    gate_sets: Optional[Iterable[serializable_gate_set.SerializableGateSet]] = None,
):
    """Creates a noiseless virtual engine object from a device specification template.

    Args:
         processor_id: name of the processor to simulate.  This is an arbitrary
             string identifier and does not have to match the processor's name
             in QCS.
         template_name: File name of the device specification template, see
             cirq_google/devices/specifications for valid templates.
         gate_sets: Iterable of serializers to use in the processor.  Defaults
             to the FSIM_GATESET.
    """
    path = pathlib.Path(__file__).parent.parent.resolve()
    f = open(path.joinpath('devices', 'specifications', template_name))
    proto_txt = f.read()
    f.close()
    device_spec = v2.device_pb2.DeviceSpecification()
    text_format.Parse(proto_txt, device_spec)
    return create_noiseless_virtual_engine_from_proto(processor_id, device_spec, gate_sets)
