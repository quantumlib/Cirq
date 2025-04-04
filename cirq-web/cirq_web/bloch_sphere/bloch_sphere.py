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

from typing import Optional

import cirq
from cirq_web import widget


class BlochSphere(widget.Widget):
    def __init__(
        self, sphere_radius: int = 5, state_vector: Optional[cirq.STATE_VECTOR_LIKE] = None
    ):
        """Initializes a BlochSphere.

        Also initializes it's parent class Widget with the bundle file provided.

        Args:
            sphere_radius: the radius of the bloch sphere in the three.js diagram.
                The default value is 5.
            state_vector: a state vector to pass in to be represented.

        Raises:
            ValueError: If the `sphere_radius` is not positive or the `state_vector` is not
                supplied.
        """
        super().__init__()
        if sphere_radius <= 0:
            raise ValueError('You must input a positive radius for the sphere')
        self.sphere_radius = sphere_radius

        if state_vector is None:
            raise ValueError('No state vector given in BlochSphere initialization')

        self.bloch_vector = cirq.bloch_vector_from_state_vector(
            cirq.to_valid_state_vector(state_vector, num_qubits=1), 0
        )

    def get_client_code(self) -> str:
        return f"""
        <script>
        renderBlochSphere('{self.id}', {self.sphere_radius})
            .addVector({self.bloch_vector[0]}, {self.bloch_vector[1]}, {self.bloch_vector[2]});
        </script>
        """

    def get_widget_bundle_name(self) -> str:
        return 'bloch_sphere.bundle.js'
