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
from typing import cast, Iterable, List, Optional, Union
import pathlib
import time

import google.protobuf.text_format as text_format
import cirq
from cirq_google.api import v2
from cirq_google.engine import calibration, engine_validator, simulated_local_processor
from cirq_google.devices import serializable_device
from cirq_google.serialization.gate_sets import FSIM_GATESET
from cirq_google.serialization import serializable_gate_set
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor

MOST_RECENT_TEMPLATES = {
    'rainbow': 'rainbow_2021_12_10_device_spec.proto.txt',
    'weber': 'weber_2021_12_10_device_spec.proto.txt',
}

MEDIAN_CALIBRATIONS = {
    'rainbow': 'rainbow_2021_11_16_calibration.json',
    'weber': 'weber_2021_11_03_calibration.json',
}

MEDIAN_CALIBRATION_TIMESTAMPS = {
    'rainbow': 1637058415838,  # 2021-11-16 10:26:55.838 UTC
    'weber': 1635923188204,  # 2021-11-03 07:06:28.204 UTC
}

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


def _create_perfect_calibration(device: cirq.Device) -> calibration.Calibration:
    all_metrics: calibration.ALL_METRICS = {}
    if device.metadata is None:
        raise ValueError('Devices for noiseless Virtual Engine must have qubits')
    qubit_set = device.metadata.qubit_set
    qubits = [cast(cirq.GridQubit, q) for q in qubit_set]
    for name in METRICS_1Q:
        all_metrics[name] = {(q,): [0.0] for q in qubits}
    for name in METRICS_2Q:
        metric_dict: calibration.METRIC_DICT = {}
        if device.metadata is not None:
            for pair in device.metadata.nx_graph.edges():
                qubits = [cast(cirq.GridQubit, q) for q in pair]
                metric_dict[(qubits[0], qubits[1])] = [0.0]
        all_metrics[name] = metric_dict
    all_metrics[T1_METRIC_NAME] = {(q,): [PERFECT_T1_VALUE] for q in qubits}
    snapshot = v2.metrics_pb2.MetricsSnapshot()
    snapshot.timestamp_ms = int(time.time() * 1000)
    return calibration.Calibration(calibration=snapshot, metrics=all_metrics)


def load_median_device_calibration(processor_id: str) -> calibration.Calibration:
    """Loads a median `cirq_google.Calibration` for the given device.

    Real calibration data from Google's 'rainbow' and 'weber' devices has been
    saved in Cirq. The calibrations selected are roughly representative of the
    median performance for that chip.

    A description of the stored metrics can be found on the
    [calibration page](https://quantumai.google/cirq/google/calibration).

    Args:
        processor_id: name of the processor to simulate.

    Raises:
        ValueError: if processor_id is not a supported QCS processor.
    """
    cal_name = MEDIAN_CALIBRATIONS.get(processor_id, None)
    if cal_name is None:
        raise ValueError(
            f"Got processor_id={processor_id}, but no median calibration "
            "is defined for that processor."
        )
    path = pathlib.Path(__file__).parent.parent.resolve()
    with path.joinpath('devices', 'calibrations', cal_name).open() as f:
        cal = cast(calibration.Calibration, cirq.read_json(f))
    cal.timestamp = MEDIAN_CALIBRATION_TIMESTAMPS[processor_id]
    return cal


def _create_virtual_processor_from_device(
    processor_id: str, device: cirq.Device
) -> simulated_local_processor.SimulatedLocalProcessor:
    """Creates a Processor object that is backed by a noiseless simulator.

    Creates a noiseless `AbstractProcessor` object based on the cirq simulator,
    a default validator, and a provided device.

    Args:
        processor_id: name of the processor to simulate.  This is an arbitrary
            string identifier and does not have to match the processor's name
            in QCS.
        device: A `cirq.Device` to validate circuits against.
    """
    calibration = _create_perfect_calibration(device)
    return simulated_local_processor.SimulatedLocalProcessor(
        processor_id=processor_id,
        device=device,
        validator=engine_validator.create_engine_validator(),
        gate_set_validator=engine_validator.create_gate_set_validator(),
        calibrations={calibration.timestamp // 1000: calibration},
    )


def create_noiseless_virtual_engine_from_device(
    processor_id: str, device: cirq.Device
) -> SimulatedLocalEngine:
    """Creates an Engine object with a single processor backed by a noiseless simulator.

    Creates a noiseless engine object based on the cirq simulator,
    a default validator, and a provided device.

    Args:
        processor_id: name of the processor to simulate.  This is an arbitrary
            string identifier and does not have to match the processor's name
            in QCS.
        device: A `cirq.Device` to validate circuits against.
    """
    return SimulatedLocalEngine([_create_virtual_processor_from_device(processor_id, device)])


def create_noiseless_virtual_processor_from_proto(
    processor_id: str,
    device_specification: v2.device_pb2.DeviceSpecification,
    gate_sets: Optional[Iterable[serializable_gate_set.SerializableGateSet]] = None,
) -> SimulatedLocalProcessor:
    """Creates a simulated local processor from a device specification proto.

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
    processor = _create_virtual_processor_from_device(processor_id, device)
    return processor


def create_noiseless_virtual_engine_from_proto(
    processor_ids: Union[str, List[str]],
    device_specifications: Union[
        v2.device_pb2.DeviceSpecification, List[v2.device_pb2.DeviceSpecification]
    ],
    gate_sets: Optional[Iterable[serializable_gate_set.SerializableGateSet]] = None,
) -> SimulatedLocalEngine:
    """Creates a noiseless virtual engine object from a device specification proto.

    The device specification protocol buffer specifies qubits and gates on the device
    and can be retrieved from a stored "proto.txt" file or from the QCS API.

    Args:
        processor_ids: names of the processors to simulate.  These are arbitrary
            string identifiers and do not have to match the processors' names
            in QCS.  This can be a single string or list of strings.
        device_specifications:  `v2.device_pb2.DeviceSpecification` proto to create
            validating devices from.  This can be a single DeviceSpecification
            or a list of them.  There should be one DeviceSpecification for each
            processor_id.
        gate_sets: Iterable of serializers to use in the processor.  Defaults
            to the FSIM_GATESET.

    Raises:
        ValueError: if processor_ids and device_specifications are not the same length.
    """
    if gate_sets is None:
        gate_sets = [FSIM_GATESET]
    if isinstance(processor_ids, str):
        processor_ids = [processor_ids]
    if isinstance(device_specifications, v2.device_pb2.DeviceSpecification):
        device_specifications = [device_specifications]
    if len(processor_ids) != len(device_specifications):
        raise ValueError('Must provide equal numbers of processor ids and device specifications.')

    return SimulatedLocalEngine(
        processors=[
            create_noiseless_virtual_processor_from_proto(processor_id, device_spec, gate_sets)
            for device_spec, processor_id in zip(device_specifications, processor_ids)
        ]
    )


def _create_device_spec_from_template(template_name: str) -> v2.device_pb2.DeviceSpecification:
    """Load a template proto into a `v2.device_pb2.DeviceSpecification`."""

    path = pathlib.Path(__file__).parent.parent.resolve()
    with path.joinpath('devices', 'specifications', template_name).open() as f:
        proto_txt = f.read()
    device_spec = v2.device_pb2.DeviceSpecification()
    text_format.Parse(proto_txt, device_spec)
    return device_spec


def create_device_from_processor_id(processor_id: str) -> serializable_device.SerializableDevice:
    """Generates a `cirq_google.SerializableDevice` for a given processor ID.

    Args:
        processor_id: name of the processor to simulate.

    Raises:
        ValueError: if processor_id is not a supported QCS processor.
    """
    template_name = MOST_RECENT_TEMPLATES.get(processor_id, None)
    if template_name is None:
        raise ValueError(f"Got processor_id={processor_id}, but no such processor is defined.")
    device_specification = _create_device_spec_from_template(template_name)
    return serializable_device.SerializableDevice.from_proto(device_specification, [FSIM_GATESET])


def create_noiseless_virtual_processor_from_template(
    processor_id: str,
    template_name: str,
    gate_sets: Optional[Iterable[serializable_gate_set.SerializableGateSet]] = None,
) -> SimulatedLocalProcessor:
    """Creates a simulated local processor from a device specification template.

    Args:
        processor_id: name of the processor to simulate.  This is an arbitrary
            string identifier and does not have to match the processor's name
            in QCS.
        template_name: File name of the device specification template, see
            cirq_google/devices/specifications for valid templates.
        gate_sets: Iterable of serializers to use in the processor.  Defaults
            to the FSIM_GATESET.
    """
    return create_noiseless_virtual_processor_from_proto(
        processor_id,
        device_specification=_create_device_spec_from_template(template_name),
        gate_sets=gate_sets,
    )


def create_noiseless_virtual_engine_from_templates(
    processor_ids: Union[str, List[str]],
    template_names: Union[str, List[str]],
    gate_sets: Optional[Iterable[serializable_gate_set.SerializableGateSet]] = None,
) -> SimulatedLocalEngine:
    """Creates a noiseless virtual engine object from a device specification template.

    Args:
        processor_ids: names of the processors to simulate.  These are arbitrary
            string identifiers and do not have to match the processors' names
            in QCS.  There can be a single string or a list of strings for multiple
            processors.
        template_names: File names of the device specification templates, see
            cirq_google/devices/specifications for valid templates.  There can
            be a single str for a template name or a list of strings.  Each
            template name should be matched to a single processor id.
        gate_sets: Iterable of serializers to use in the processor.  Defaults
            to the FSIM_GATESET.

    Raises:
        ValueError: if processor_ids and template_names are not the same length.
    """
    if isinstance(processor_ids, str):
        processor_ids = [processor_ids]
    if isinstance(template_names, str):
        template_names = [template_names]
    if len(processor_ids) != len(template_names):
        raise ValueError('Must provide equal numbers of processor ids and template names.')

    specifications = [
        _create_device_spec_from_template(template_name) for template_name in template_names
    ]
    return create_noiseless_virtual_engine_from_proto(processor_ids, specifications, gate_sets)


def create_noiseless_virtual_engine_from_latest_templates() -> SimulatedLocalEngine:
    """Creates a noiseless virtual engine based on current templates.

    This uses the most recent templates to create a reasonable facsimile of
    a simulated Quantum Computing Service (QCS).

    Note:  this will use the most recent templates to match the service.
    While not expected to change frequently, this function may change the
    templates (processors) that are included in the "service" as the actual
    hardware evolves.  The processors returned from this function should not
    be considered stable from version to version and are not guaranteed to be
    backwards compatible.
    """
    processor_ids = list(MOST_RECENT_TEMPLATES.keys())
    template_names = [MOST_RECENT_TEMPLATES[k] for k in processor_ids]
    return create_noiseless_virtual_engine_from_templates(processor_ids, template_names)
