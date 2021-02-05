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

from cirq.google.calibration.phased_fsim import PhasedFSimCharacterization


def test_asdict():
    characterization_angles = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    characterization = PhasedFSimCharacterization(**characterization_angles)
    assert characterization.asdict() == characterization_angles


def test_all_none():
    assert PhasedFSimCharacterization().all_none()

    characterization_angles = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    for angle, value in characterization_angles.items():
        assert not PhasedFSimCharacterization(**{angle: value}).all_none()


def test_any_none():
    characterization_angles = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    assert not PhasedFSimCharacterization(**characterization_angles).any_none()

    for angle in characterization_angles:
        none_angles = dict(characterization_angles)
        del none_angles[angle]
        assert PhasedFSimCharacterization(**none_angles).any_none()


def test_parameters_for_qubits_swapped():
    characterization = PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5)
    assert characterization.parameters_for_qubits_swapped() == PhasedFSimCharacterization(
        theta=0.1, zeta=-0.2, chi=-0.3, gamma=0.4, phi=0.5
    )


def test_merge_with():
    characterization = PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3)
    other = PhasedFSimCharacterization(gamma=0.4, phi=0.5, theta=0.6)
    assert characterization.merge_with(other) == PhasedFSimCharacterization(
        theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5
    )


def test_override_by():
    characterization = PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3)
    other = PhasedFSimCharacterization(gamma=0.4, phi=0.5, theta=0.6)
    assert characterization.override_by(other) == PhasedFSimCharacterization(
        theta=0.6, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5
    )
