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

import functools
from typing import Dict

from cirq.protocols.json_serialization import ObjectFactory


@functools.lru_cache(maxsize=1)
def _class_resolver_dictionary() -> Dict[str, ObjectFactory]:
    import cirq.google
    from cirq.google.devices.known_devices import _NamedConstantXmonDevice

    return {
        '_NamedConstantXmonDevice': _NamedConstantXmonDevice,
        'Calibration': cirq.google.Calibration,
        'CalibrationTag': cirq.google.CalibrationTag,
        'CalibrationLayer': cirq.google.CalibrationLayer,
        'CalibrationResult': cirq.google.CalibrationResult,
        'SycamoreGate': cirq.google.SycamoreGate,
        'GateTabulation': cirq.google.GateTabulation,
        'PhysicalZTag': cirq.google.PhysicalZTag,
        'FloquetPhasedFSimCalibrationOptions': cirq.google.FloquetPhasedFSimCalibrationOptions,
        'FloquetPhasedFSimCalibrationRequest': cirq.google.FloquetPhasedFSimCalibrationRequest,
        'PhasedFSimCalibrationResult': cirq.google.PhasedFSimCalibrationResult,
        'PhasedFSimCharacterization': cirq.google.PhasedFSimCharacterization,
    }
