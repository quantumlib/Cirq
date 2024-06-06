# Copyright 2024 Scaleway
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

from cirq_scaleway import ScalewayQuantumService

service = ScalewayQuantumService(
    project_id="<your-scaleway-project-id>", secret_key="<your-scaleway-secret-key>"
)

# Get and display all provided (real or simulated) devices
devices = service.devices()
print(devices)

# Create a sampler from the a device name
sampler = service.sampler(device="qsim_simulation_local")

qubit = cirq.GridQubit(0, 0)
circuit = cirq.Circuit(cirq.X(qubit) ** 0.5, cirq.measure(qubit, key='m'))

# Run the circuit on the device
result = sampler.run(circuit)
print(result)
