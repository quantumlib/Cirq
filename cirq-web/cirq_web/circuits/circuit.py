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

from cirq_web import widget

class Circuit3D(widget.Widget):
    def __init__(self):
        """Initializes a Circuit instance.
        
        """
        super().__init__()
        pass

    def get_client_code(self) -> str:
        return f"""
            <div id="test">
            <script>
            const circuit = createGridCircuit(5);
            circuit.addCube(1, 0, 1); 
            </script>
        """

    def get_widget_bundle_name(self) -> str:
        return 'circuit.bundle.js'
