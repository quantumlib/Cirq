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

from __future__ import annotations

import unittest.mock as mock

import google.protobuf.text_format as text_format
import numpy as np
import pytest

import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory


def _test_processor(processor: cg.engine.abstract_processor.AbstractProcessor) -> None:
    """Tests an engine instance with some standard commands.
    Also tests the non-Sycamore qubits and gates fail."""
    good_qubit = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(good_qubit), cirq.measure(good_qubit))
    results = processor.run(circuit, repetitions=100, run_name="run", device_config_name="config")
    assert np.all(results.measurements[str(good_qubit)] == 1)
    with pytest.raises(RuntimeError, match='requested total repetitions'):
        _ = processor.run(
            circuit, repetitions=100_000_000, run_name="run", device_config_name="config"
        )

    bad_qubit = cirq.GridQubit(1000, 1000)
    circuit = cirq.Circuit(cirq.X(bad_qubit), cirq.measure(bad_qubit))
    with pytest.raises(ValueError, match='Qubit not on device'):
        _ = processor.run(circuit, repetitions=100, run_name="run", device_config_name="config")
    circuit = cirq.Circuit(
        cirq.testing.DoesNotSupportSerializationGate()(good_qubit), cirq.measure(good_qubit)
    )
    with pytest.raises(ValueError, match='Cannot serialize op'):
        _ = processor.run(circuit, repetitions=100, run_name="run", device_config_name="config")


def test_create_device_from_processor_id() -> None:
    device = factory.create_device_from_processor_id('rainbow')
    assert device is not None

    with pytest.raises(ValueError, match='no such processor is defined'):
        _ = factory.create_device_from_processor_id('bad_processor')


def test_create_from_device() -> None:
    engine = factory.create_noiseless_virtual_engine_from_device('sycamore', cg.Sycamore)
    _test_processor(engine.get_processor('sycamore'))


def test_load_device_noise_properties() -> None:
    noise_properties = factory.load_device_noise_properties("rainbow")
    assert noise_properties.readout_errors[cirq.GridQubit(9, 4)] == [0.0124, 0.0464]
    assert noise_properties.gate_times_ns[cirq.PhasedXZGate] == 25
    op_id = cirq.OpIdentifier(cirq.ISwapPowGate, cirq.GridQubit(3, 2), cirq.GridQubit(4, 2))
    assert noise_properties.fsim_errors[op_id].zeta == -0.004952147720840733
    assert noise_properties.fsim_errors[op_id].gamma == -0.04094895320428251


def test_median_rainbow_device() -> None:
    q0, q1 = cirq.GridQubit.rect(1, 2, 5, 3)
    cal = factory.load_median_device_calibration('rainbow')
    # Spot-check an arbitrary set of values to confirm we got the right data.
    assert np.isclose(cal['parallel_p00_error'][(q0,)][0], 0.0067149999999998045)
    assert np.isclose(
        cal['single_qubit_readout_separation_error'][(q1,)][0], 0.00039635847369929797
    )
    assert np.isclose(
        cal['two_qubit_sycamore_gate_xeb_average_error_per_cycle'][(q0, q1)][0],
        0.0034558565201338043,
    )


def test_median_weber_device() -> None:
    q0, q1 = cirq.GridQubit.rect(1, 2, 1, 4)
    cal = factory.load_median_device_calibration('weber')
    # Spot-check an arbitrary set of values to confirm we got the right data.
    assert np.isclose(cal['parallel_p00_error'][(q0,)][0], 0.00592)
    assert np.isclose(cal['single_qubit_readout_separation_error'][(q1,)][0], 0.0005456228122027591)
    assert np.isclose(
        cal['two_qubit_sycamore_gate_xeb_average_error_per_cycle'][(q0, q1)][0],
        0.010057137642549896,
    )


@pytest.mark.parametrize('processor_id', ['rainbow', 'weber'])
def test_median_device_expected_fields(processor_id: str) -> None:
    cal = factory.load_median_device_calibration(processor_id)
    expected_fields = {
        'single_qubit_idle_t1_micros',
        'single_qubit_rb_incoherent_error_per_gate',
        'single_qubit_rb_pauli_error_per_gate',
        'two_qubit_parallel_sycamore_gate_xeb_pauli_error_per_cycle',
        'two_qubit_parallel_sqrt_iswap_gate_xeb_pauli_error_per_cycle',
        'single_qubit_p00_error',
        'single_qubit_p11_error',
        'two_qubit_parallel_sycamore_gate_xeb_entangler_theta_error_per_cycle',
        'two_qubit_parallel_sycamore_gate_xeb_entangler_phi_error_per_cycle',
        'two_qubit_parallel_sqrt_iswap_gate_xeb_entangler_theta_error_per_cycle',
        'two_qubit_parallel_sqrt_iswap_gate_xeb_entangler_phi_error_per_cycle',
    }
    assert expected_fields.issubset(cal.keys())


def test_median_device_expected_fields_willow_pink() -> None:
    cal = factory.load_median_device_calibration("willow_pink")
    expected_fields = {
        'single_qubit_idle_t1_micros',
        'single_qubit_rb_incoherent_error_per_gate',
        'single_qubit_rb_pauli_error_per_gate',
        'two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle',
        'single_qubit_p00_error',
        'single_qubit_p11_error',
        'two_qubit_parallel_cz_gate_xeb_entangler_theta_error_per_cycle',
        'two_qubit_parallel_cz_gate_xeb_entangler_phi_error_per_cycle',
    }
    assert expected_fields.issubset(cal.keys())


def test_median_device_bad_processor() -> None:
    with pytest.raises(ValueError, match='no median calibration is defined'):
        _ = factory.load_median_device_calibration('bad_processor')


@pytest.mark.parametrize('processor_id', ['rainbow', 'weber'])
def test_sample_device_zphase(processor_id: str) -> None:
    zphase_data = factory.load_sample_device_zphase(processor_id)
    assert 'sqrt_iswap' in zphase_data
    sqrt_iswap_data = zphase_data['sqrt_iswap']
    for angle in ['zeta', 'gamma']:
        assert angle in sqrt_iswap_data
        for (q0, q1), val in sqrt_iswap_data[angle].items():
            assert isinstance(q0, cirq.Qid)
            assert isinstance(q1, cirq.Qid)
            assert isinstance(val, float)


def test_device_zphase_bad_processor() -> None:
    with pytest.raises(ValueError, match='no Z phase data is defined'):
        _ = factory.load_sample_device_zphase('bad_processor')


def test_create_from_proto() -> None:
    # Create a minimal gate specification that can handle the test.
    device_spec = text_format.Merge(
        """
valid_qubits: "5_4"
valid_gates {
  phased_xz {}
}
valid_gates {
  meas {}
}
""",
        v2.device_pb2.DeviceSpecification(),
    )
    engine = factory.create_noiseless_virtual_engine_from_proto('sycamore', device_spec)
    _test_processor(engine.get_processor('sycamore'))
    assert engine.get_processor('sycamore').get_device_specification() == device_spec

    processor = factory.create_noiseless_virtual_processor_from_proto('sycamore', device_spec)
    _test_processor(processor)
    assert processor.get_device_specification() == device_spec


def test_create_from_template() -> None:
    engine = factory.create_noiseless_virtual_engine_from_templates(
        'sycamore', 'weber_2021_12_10_device_spec_for_grid_device.proto.txt'
    )
    _test_processor(engine.get_processor('sycamore'))

    processor = factory.create_noiseless_virtual_processor_from_template(
        'sycamore', 'weber_2021_12_10_device_spec_for_grid_device.proto.txt'
    )
    _test_processor(processor)


@pytest.mark.parametrize('processor_id', ['rainbow', 'weber', 'willow_pink'])
def test_default_creation(processor_id: str) -> None:
    engine = factory.create_noiseless_virtual_engine_from_latest_templates()
    _test_processor(engine.get_processor(processor_id))
    processor = engine.get_processor(processor_id)
    device_specification = processor.get_device_specification()
    expected = factory.create_device_spec_from_processor_id(processor_id)
    assert device_specification is not None
    assert device_specification == expected


def test_create_from_template_wrong_args() -> None:
    with pytest.raises(ValueError, match='equal numbers of processor ids'):
        _ = factory.create_noiseless_virtual_engine_from_templates(
            ['sycamore', 'sycamore2'], 'weber_2021_12_10_device_spec.proto.txt'
        )
    with pytest.raises(ValueError, match='equal numbers of processor ids'):
        _ = factory.create_noiseless_virtual_engine_from_proto('sycamore', [])


def test_create_from_proto_no_qubits() -> None:
    with pytest.raises(ValueError, match='must have qubits'):
        _ = factory.create_noiseless_virtual_engine_from_device(
            'sycamore', cirq.UNCONSTRAINED_DEVICE
        )


@pytest.mark.parametrize(
    'processor_id,extract_gate_times_call_count', [('rainbow', 0), ('weber', 0), ('willow_pink', 1)]
)
def test_create_default_noisy_quantum_virtual_machine(
    processor_id: str, extract_gate_times_call_count: int
) -> None:
    engine = factory.create_default_noisy_quantum_virtual_machine(
        processor_id=processor_id, simulator_class=cirq.Simulator
    )
    processor = engine.get_processor(processor_id)
    bad_qubit = cirq.GridQubit(1000, 1000)
    circuit = cirq.Circuit(cirq.X(bad_qubit), cirq.measure(bad_qubit))
    with pytest.raises(ValueError, match='Qubit not on device'):
        _ = processor.run(circuit, repetitions=100, run_name="run", device_config_name="config")
    good_qubit = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(
        cirq.testing.DoesNotSupportSerializationGate()(good_qubit), cirq.measure(good_qubit)
    )
    with pytest.raises(ValueError, match='.* contains a gate which is not supported.'):
        _ = processor.run(circuit, repetitions=100, run_name="run", device_config_name="config")
    device_specification = processor.get_device_specification()
    expected = factory.create_device_spec_from_processor_id(processor_id)
    assert device_specification is not None
    assert device_specification == expected
    with mock.patch.object(
        factory, "extract_gate_times_ns_from_device", return_value=None
    ) as mocked_extract:
        _ = factory.create_default_noisy_quantum_virtual_machine(
            processor_id=processor_id, simulator_class=cirq.Simulator
        )
        assert mocked_extract.call_count == extract_gate_times_call_count


def test_extract_gate_times_ns_from_device() -> None:
    device = factory.create_device_from_processor_id('willow_pink')
    gate_times_ns = factory.extract_gate_times_ns_from_device(device)
    assert gate_times_ns[cirq.CZPowGate] == 42
    assert gate_times_ns[cirq.MeasurementGate] == 600
    assert gate_times_ns[cirq.XPowGate] == 25
    assert gate_times_ns[cirq.ResetChannel] == 160
    assert cirq.IdentityGate not in gate_times_ns
    assert cirq.WaitGate not in gate_times_ns


def test_extract_gate_times_ns_from_device_without_durations() -> None:
    metadata_without_durations = cirq.GridDeviceMetadata(
        qubit_pairs=[(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))],
        gateset=cirq.Gateset(cirq.XPowGate),
    )
    device_without_durations = cg.GridDevice(metadata_without_durations)
    assert factory.extract_gate_times_ns_from_device(device_without_durations) == {}
