# Copyright 2020 The Cirq Developers
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

from cirq_ionq._version import __version__

from cirq_ionq.calibration import Calibration

from cirq_ionq.ionq_devices import IonQAPIDevice, decompose_to_device

from cirq_ionq.ionq_gateset import IonQTargetGateset

from cirq_ionq.ionq_exceptions import (
    IonQException,
    IonQNotFoundException,
    IonQUnsuccessfulJobException,
)

from cirq_ionq.job import Job

from cirq_ionq.results import QPUResult, SimulatorResult

from cirq_ionq.sampler import Sampler

from cirq_ionq.serializer import Serializer, SerializedProgram

from cirq_ionq.service import Service

from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate

from cirq.protocols.json_serialization import _register_resolver
from cirq_ionq.json_resolver_cache import _class_resolver_dictionary

_register_resolver(_class_resolver_dictionary)
