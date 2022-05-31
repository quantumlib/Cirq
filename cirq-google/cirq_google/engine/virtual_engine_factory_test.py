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
import pytest
import numpy as np
import cirq
import cirq_google as cg
import cirq_google.api.v2 as v2
import cirq_google.engine.virtual_engine_factory as factory


def _test_processor(processor: cg.engine.abstract_processor.AbstractProcessor):
    """Tests an engine instance with some standard commands.
    Also tests the non-Sycamore qubits and gates fail."""
    good_qubit = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(good_qubit), cirq.measure(good_qubit))
    results = processor.run(circuit, repetitions=100)
    assert np.all(results.measurements[str(good_qubit)] == 1)
    with pytest.raises(RuntimeError, match='requested total repetitions'):
        _ = processor.run(circuit, repetitions=100_000_000)

    bad_qubit = cirq.GridQubit(10, 10)
    circuit = cirq.Circuit(cirq.X(bad_qubit), cirq.measure(bad_qubit))
    with pytest.raises(ValueError, match='Qubit not on device'):
        _ = processor.run(circuit, repetitions=100)
    circuit = cirq.Circuit(cirq.H(good_qubit), cirq.measure(good_qubit))
    with pytest.raises(ValueError, match='Cannot serialize op'):
        _ = processor.run(circuit, repetitions=100)


def test_create_device_from_processor_id():
    device = factory.create_device_from_processor_id('rainbow')
    assert device is not None

    with pytest.raises(ValueError, match='no such processor is defined'):
        _ = factory.create_device_from_processor_id('bad_processor')


def test_create_from_device():
    engine = factory.create_noiseless_virtual_engine_from_device('sycamore', cg.Sycamore)
    _test_processor(engine.get_processor('sycamore'))


def test_median_rainbow_device():
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


def test_median_weber_device():
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
def test_median_device_expected_fields(processor_id):
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


def test_median_device_bad_processor():
    with pytest.raises(ValueError, match='no median calibration is defined'):
        _ = factory.load_median_device_calibration('bad_processor')


def test_create_from_proto():

    # Create a minimal gate specification that can handle the test.
    device_spec = v2.device_pb2.DeviceSpecification()
    device_spec.valid_qubits.extend(['5_4'])
    gs = device_spec.valid_gate_sets.add()
    gs.name = 'fsim'
    gs.valid_gates.add().id = 'fsim'
    gs.valid_gates.add().id = 'xyz'
    gs.valid_gates.add().id = 'xy'
    gs.valid_gates.add().id = 'z'
    gs.valid_gates.add().id = 'meas'
    gs.valid_gates.add().id = 'wait'
    gs.valid_gates.add().id = 'circuit'
    engine = factory.create_noiseless_virtual_engine_from_proto('sycamore', device_spec)
    _test_processor(engine.get_processor('sycamore'))

    processor = factory.create_noiseless_virtual_processor_from_proto('sycamore', device_spec)
    _test_processor(processor)


def test_create_from_template():
    engine = factory.create_noiseless_virtual_engine_from_templates(
        'sycamore', 'weber_2021_12_10_device_spec.proto.txt'
    )
    _test_processor(engine.get_processor('sycamore'))

    processor = factory.create_noiseless_virtual_processor_from_template(
        'sycamore', 'weber_2021_12_10_device_spec.proto.txt'
    )
    _test_processor(processor)


def test_default_creation():
    engine = factory.create_noiseless_virtual_engine_from_latest_templates()
    _test_processor(engine.get_processor('weber'))
    _test_processor(engine.get_processor('rainbow'))


def test_create_from_template_wrong_args():
    with pytest.raises(ValueError, match='equal numbers of processor ids'):
        _ = factory.create_noiseless_virtual_engine_from_templates(
            ['sycamore', 'sycamore2'], 'weber_2021_12_10_device_spec.proto.txt'
        )
    with pytest.raises(ValueError, match='equal numbers of processor ids'):
        _ = factory.create_noiseless_virtual_engine_from_proto('sycamore', [])


def test_create_from_proto_no_qubits():
    with pytest.raises(ValueError, match='must have qubits'):
        _ = factory.create_noiseless_virtual_engine_from_device(
            'sycamore', cirq.UNCONSTRAINED_DEVICE
        )
