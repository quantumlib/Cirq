# Copyright 2018 The Cirq Developers
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
from typing import Any, Iterable
import numpy as np
import pytest

import cirq


def make_buffers(shape: Iterable[int], dtype: np.dtype) -> Iterable[np.ndarray]:
    return (np.empty(shape, dtype=dtype), np.empty(shape, dtype=dtype),
            np.empty(shape, dtype=dtype))


def apply_channel(val: Any, rho: np.ndarray) -> np.ndarray:
    out_buf, buf0, buf1 = make_buffers(rho.shape, rho.dtype)
    result = cirq.apply_channel(val,
                                args=cirq.ApplyChannelArgs(
                                    target_tensor=rho,
                                    left_axes=[0],
                                    right_axes=[1],
                                    out_buffer=out_buf,
                                    auxiliary_buffer0=buf0,
                                    auxiliary_buffer1=buf1))
    return result


def assert_same_effect(rho: np.ndarray, a: Any, b: Any) -> None:
    res1 = apply_channel(a, np.copy(rho))
    res2 = apply_channel(b, np.copy(rho))

    np.testing.assert_array_almost_equal(res1, res2)


def test_thermal_relaxation_recovers_amplitude_damping():
    for a in np.linspace(0, 1, 15):
        for b in np.linspace(0, 1, 15):
            tr = cirq.thermal_relaxation(a, b, b)

            tr2 = cirq.generalized_amplitude_damp(a, b)

            state = cirq.testing.random_superposition(2)
            rho = np.outer(state, state)

            assert_same_effect(rho, tr2, tr)


def test_thermal_relaxation_recovers_phase_damping():
    for a in np.linspace(0, 1, 15):
        tr = cirq.thermal_relaxation(1.0, 0.0, a)

        tr2 = cirq.phase_damp(a)

        state = cirq.testing.random_superposition(2)
        rho = np.outer(state, state)

        assert_same_effect(rho, tr2, tr)


def test_thermal_relaxation_simultaneous_decay_diagonal():
    rho = np.eye(2) * 0.5

    ch = cirq.thermal_relaxation(0.95, 0.1, 0.3)
    res = apply_channel(ch, rho)
    expected = np.array([[0.545, 0], [0, 0.455]])

    np.testing.assert_array_almost_equal(res, expected)


def test_thermal_relaxation_simultaneous_decay_full():
    rho = np.array([[0.5, 0.5], [0.5, 0.5]])

    ch = cirq.thermal_relaxation(0.95, 0.1, 0.3)
    res = apply_channel(ch, rho)
    expected = np.array([[0.545, 0.41833], [0.41833, 0.455]])

    np.testing.assert_array_almost_equal(res, expected)


def test_thermal_relaxation_has_channel():
    assert cirq.has_channel(cirq.ThermalRelaxationChannel(0.1, 0.2, 0.3))


def test_thermal_relaxation_properties():
    ch = cirq.ThermalRelaxationChannel(0.1, 0.2, 0.3)
    assert ch.p_exchange == 0.1
    assert ch.p_relaxation == 0.2
    assert ch.p_dephasing == 0.3


def test_thermal_relaxation_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.ThermalRelaxationChannel(0.1, 0.2, 0.3))


def test_thermal_relaxation_channel_str():
    assert (str(cirq.thermal_relaxation(0.1, 0.2, 0.3)) == \
        'thermal_relaxation(p_exchange=0.1,p_relaxation=0.2,p_dephasing=0.3)')


def test_thermal_relaxation_text_diagram():
    round_to_2_prec = cirq.CircuitDiagramInfoArgs(known_qubits=None,
                                                  known_qubit_count=None,
                                                  use_unicode_characters=True,
                                                  precision=2,
                                                  qubit_map=None)

    none_prec = cirq.CircuitDiagramInfoArgs(known_qubits=None,
                                            known_qubit_count=None,
                                            use_unicode_characters=True,
                                            precision=None,
                                            qubit_map=None)

    a = cirq.thermal_relaxation(0.123, 0.456, 0.789)
    assert (cirq.circuit_diagram_info(
        a, args=round_to_2_prec) == cirq.CircuitDiagramInfo(
            wire_symbols=('ThR(0.12,0.46,0.79)',)))
    assert (cirq.circuit_diagram_info(
        a, args=none_prec) == cirq.CircuitDiagramInfo(
            wire_symbols=('ThR(0.123,0.456,0.789)',)))


def test_thermal_relaxation_invalid_probs():
    with pytest.raises(ValueError, match='p_exchange'):
        cirq.thermal_relaxation(-5, 0.5, 0.5)

    with pytest.raises(ValueError, match='p_relaxation'):
        cirq.thermal_relaxation(0.5, -5, 0.5)

    with pytest.raises(ValueError, match='p_dephasing'):
        cirq.thermal_relaxation(0.5, 0.5, -5)


def test_thermal_relaxation_non_cp():
    with pytest.raises(ValueError, match='CP requirement.'):
        cirq.thermal_relaxation(0.1, 0.9, 0.3)

    with pytest.raises(ValueError, match='CP requirement.'):
        cirq.thermal_relaxation(0.9, 0.5, 0.4)
