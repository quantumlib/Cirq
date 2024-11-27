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

"""Qubit Gates, Operations, and Tags useful for Google devices. """

from cirq_google.ops.calibration_tag import CalibrationTag as CalibrationTag

from cirq_google.ops.coupler import Coupler as Coupler

from cirq_google.ops.fsim_gate_family import FSimGateFamily as FSimGateFamily

from cirq_google.ops.fsim_via_model_tag import FSimViaModelTag as FSimViaModelTag

from cirq_google.ops.physical_z_tag import PhysicalZTag as PhysicalZTag

from cirq_google.ops.sycamore_gate import SycamoreGate as SycamoreGate, SYC as SYC

from cirq_google.ops.internal_gate import InternalGate as InternalGate

from cirq_google.ops.dynamical_decoupling_tag import (
    DynamicalDecouplingTag as DynamicalDecouplingTag,
)
