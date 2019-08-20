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


class ParamValue():

    # range for numeric types
    # discrete

class GateParam():

    def __init__(self, attr: str, values):
        pass


class GateSpec():

    def __init__(self, constructor, params):
        pass

    def canonical

class GateSet():

    def __init__(self, gate_specs):
        pass

    def is_supported_gate(self, gate) -> bool:
        # checks that this gate is valid for this gate set
        # includes checking that the gates parameter values are valid.
        pass


    def union(self):
