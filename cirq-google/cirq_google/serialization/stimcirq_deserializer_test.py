# Copyright 2025 The Cirq Developers
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

import pytest

from cirq_google.api import v2
from cirq_google.serialization.stimcirq_deserializer import StimCirqDeserializer


def test_bad_stimcirq_op():
    proto = v2.program_pb2.Operation()
    proto.internalgate.module = 'stimcirq'
    proto.internalgate.name = 'WolfgangPauli'

    with pytest.raises(ValueError, match='not recognized'):
        _ = StimCirqDeserializer().from_proto(proto, constants=[], deserialized_constants=[])


def test_bad_pauli_gate():
    proto = v2.program_pb2.Operation()
    proto.internalgate.module = 'stimcirq'
    proto.internalgate.name = 'SweepPauli'
    proto.internalgate.gate_args['stim_sweep_bit_index'].arg_value.float_value = 1.0
    proto.internalgate.gate_args['cirq_sweep_symbol'].arg_value.string_value = 't'
    proto.internalgate.gate_args['pauli'].arg_value.string_value = 'Q'

    with pytest.raises(ValueError, match='pauli'):
        _ = StimCirqDeserializer().from_proto(proto, constants=[], deserialized_constants=[])
