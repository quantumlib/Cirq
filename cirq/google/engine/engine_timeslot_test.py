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
import cirq
import cirq.google as cg

from cirq.google.engine.client.quantum_v1alpha1.gapic import enums


def test_timeslot_equality():
    start = 1582592400
    end = 1582596000
    eq = cirq.testing.equals_tester.EqualsTester()
    eq.add_equality_group(
        cg.EngineTimeSlot(start_seconds=start, end_seconds=end),
        cg.EngineTimeSlot(start_seconds=start, end_seconds=end),
        cg.EngineTimeSlot(start_seconds=start,
                          end_seconds=end,
                          slot_type=enums.QuantumProcessor.TimeSlotType.
                          TIME_SLOT_TYPE_UNSPECIFIED))
    eq.add_equality_group(
        cg.EngineTimeSlot(start_seconds=start,
                          end_seconds=end,
                          project_id='123456'))
    eq.add_equality_group(
        cg.EngineTimeSlot(
            start_seconds=start,
            end_seconds=end,
            slot_type=enums.QuantumProcessor.TimeSlotType.RESERVATION,
            project_id='123456'))
    eq.add_equality_group(
        cg.EngineTimeSlot(
            start_seconds=start,
            end_seconds=end,
            slot_type=enums.QuantumProcessor.TimeSlotType.MAINTENANCE,
            project_id='123456'))
    eq.add_equality_group(
        cg.EngineTimeSlot(
            start_seconds=start,
            end_seconds=end,
            slot_type=enums.QuantumProcessor.TimeSlotType.MAINTENANCE,
            project_id='123456',
            maintenance_title="Testing",
            maintenance_description="Testing some new configuration."))
