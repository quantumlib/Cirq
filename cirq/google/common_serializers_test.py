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
import cirq.google as cg


def test_common_serializers():
    """Test the common serializers by instantiating gate sets from them.
    """
    cg.serializable_gate_set.SerializableGateSet(
        gate_set_name='test_all',
        serializers=[cg.MEASUREMENT_SERIALIZER] + cg.SINGLE_QUBIT_SERIALIZERS,
        deserializers=([cg.MEASUREMENT_DESERIALIZER] +
                       cg.SINGLE_QUBIT_DESERIALIZERS))
    cg.serializable_gate_set.SerializableGateSet(
        gate_set_name='test_half_pi',
        serializers=([cg.MEASUREMENT_SERIALIZER] +
                     cg.SINGLE_QUBIT_HALF_PI_SERIALIZERS),
        deserializers=([cg.MEASUREMENT_DESERIALIZER] +
                       cg.SINGLE_QUBIT_HALF_PI_DESERIALIZERS))
