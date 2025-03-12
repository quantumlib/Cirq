# Copyright 2022 The Cirq Developers
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

"""Utilities for running end-to-end experiments using Quantum Computing Service (QCS)."""

from cirq_google.workflow.quantum_executable import (
    ExecutableSpec as ExecutableSpec,
    KeyValueExecutableSpec as KeyValueExecutableSpec,
    QuantumExecutable as QuantumExecutable,
    QuantumExecutableGroup as QuantumExecutableGroup,
    BitstringsMeasurement as BitstringsMeasurement,
)

from cirq_google.workflow.quantum_runtime import (
    SharedRuntimeInfo as SharedRuntimeInfo,
    RuntimeInfo as RuntimeInfo,
    ExecutableResult as ExecutableResult,
    ExecutableGroupResult as ExecutableGroupResult,
    QuantumRuntimeConfiguration as QuantumRuntimeConfiguration,
    execute as execute,
)

from cirq_google.workflow.io import (
    ExecutableGroupResultFilesystemRecord as ExecutableGroupResultFilesystemRecord,
)

from cirq_google.workflow.qubit_placement import (
    QubitPlacer as QubitPlacer,
    CouldNotPlaceError as CouldNotPlaceError,
    NaiveQubitPlacer as NaiveQubitPlacer,
    RandomDevicePlacer as RandomDevicePlacer,
    HardcodedQubitPlacer as HardcodedQubitPlacer,
)

from cirq_google.workflow.processor_record import (
    ProcessorRecord as ProcessorRecord,
    EngineProcessorRecord as EngineProcessorRecord,
    SimulatedProcessorRecord as SimulatedProcessorRecord,
    SimulatedProcessorWithLocalDeviceRecord as SimulatedProcessorWithLocalDeviceRecord,
)
