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

import warnings
import functools
from typing import Dict

from cirq.protocols.json_serialization import ObjectFactory


@functools.lru_cache()
def _class_resolver_dictionary() -> Dict[str, ObjectFactory]:
    def _old_xmon(*args, **kwargs):
        d_type = kwargs['constant']
        warnings.warn(
            f'Attempted to json load a {d_type} Device.'
            'These devices were removed in Cirq v0.15 and are no '
            'longer supported. Please update device usage to a '
            'supported device or downgrade your Cirq installation.'
        )
        return str(d_type)

    import cirq_google

    return {
        '_NamedConstantXmonDevice': _old_xmon,
        'Calibration': cirq_google.Calibration,
        'CalibrationTag': cirq_google.CalibrationTag,
        'CalibrationLayer': cirq_google.CalibrationLayer,
        'CalibrationResult': cirq_google.CalibrationResult,
        'CouplerPulse': cirq_google.experimental.CouplerPulse,
        'GoogleNoiseProperties': cirq_google.GoogleNoiseProperties,
        'SycamoreGate': cirq_google.SycamoreGate,
        'GateTabulation': cirq_google.GateTabulation,
        'PhysicalZTag': cirq_google.PhysicalZTag,
        'FSimGateFamily': cirq_google.FSimGateFamily,
        'FloquetPhasedFSimCalibrationOptions': cirq_google.FloquetPhasedFSimCalibrationOptions,
        'FloquetPhasedFSimCalibrationRequest': cirq_google.FloquetPhasedFSimCalibrationRequest,
        'PhasedFSimCalibrationResult': cirq_google.PhasedFSimCalibrationResult,
        'PhasedFSimCharacterization': cirq_google.PhasedFSimCharacterization,
        'SycamoreTargetGateset': cirq_google.SycamoreTargetGateset,
        'XEBPhasedFSimCalibrationOptions': cirq_google.XEBPhasedFSimCalibrationOptions,
        'XEBPhasedFSimCalibrationRequest': cirq_google.XEBPhasedFSimCalibrationRequest,
        'LocalXEBPhasedFSimCalibrationOptions': cirq_google.LocalXEBPhasedFSimCalibrationOptions,
        'LocalXEBPhasedFSimCalibrationRequest': cirq_google.LocalXEBPhasedFSimCalibrationRequest,
        'cirq.google.BitstringsMeasurement': cirq_google.BitstringsMeasurement,
        'cirq.google.QuantumExecutable': cirq_google.QuantumExecutable,
        'cirq.google.QuantumExecutableGroup': cirq_google.QuantumExecutableGroup,
        'cirq.google.KeyValueExecutableSpec': cirq_google.KeyValueExecutableSpec,
        'cirq.google.SharedRuntimeInfo': cirq_google.SharedRuntimeInfo,
        'cirq.google.RuntimeInfo': cirq_google.RuntimeInfo,
        'cirq.google.ExecutableResult': cirq_google.ExecutableResult,
        'cirq.google.ExecutableGroupResult': cirq_google.ExecutableGroupResult,
        # Pylint fights with the black formatter.
        # pylint: disable=line-too-long
        'cirq.google.ExecutableGroupResultFilesystemRecord': cirq_google.ExecutableGroupResultFilesystemRecord,
        # pylint: enable=line-too-long
        'cirq.google.QuantumRuntimeConfiguration': cirq_google.QuantumRuntimeConfiguration,
        'cirq.google.NaiveQubitPlacer': cirq_google.NaiveQubitPlacer,
        'cirq.google.RandomDevicePlacer': cirq_google.RandomDevicePlacer,
        'cirq.google.EngineProcessorRecord': cirq_google.EngineProcessorRecord,
        'cirq.google.SimulatedProcessorRecord': cirq_google.SimulatedProcessorRecord,
        # pylint: disable=line-too-long
        'cirq.google.SimulatedProcessorWithLocalDeviceRecord': cirq_google.SimulatedProcessorWithLocalDeviceRecord,
        'cirq.google.HardcodedQubitPlacer': cirq_google.HardcodedQubitPlacer,
        # pylint: enable=line-too-long
        'cirq.google.EngineResult': cirq_google.EngineResult,
        'cirq.google.GridDevice': cirq_google.GridDevice,
    }
