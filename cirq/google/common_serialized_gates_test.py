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

import cirq
import cirq.google as cg
import cirq.google.common_serialized_gates as cgc


def assert_arglists_equal(arg_list1, arg_list2):
    """Special handling for testing equality of SerializableArg.

    Since the gate_getter may be a lambda function, we need to test its
    equality in a special way by calling the function once.  Most lambdas
    should be constant functions, so this should work.  If that assumption is
    not valid in the future, this needs to be modified.
    """
    assert len(arg_list1) == len(arg_list2)
    for idx in range(len(arg_list1)):
        if (callable(arg_list1[idx].gate_getter)):
            assert callable(arg_list1[idx].gate_getter)
            assert arg_list1[idx].gate_getter(0.0) == arg_list2[
                idx].gate_getter(0.0)
            assert arg_list1[idx].serialized_name == arg_list2[
                idx].serialized_name
            assert arg_list1[idx].serialized_type == arg_list2[
                idx].serialized_type
        else:
            assert (arg_list1[idx] == arg_list2[idx])


def assert_serializers_equal(serializer1, serializer2):
    """Simple value equality check for op_serializer """
    assert_arglists_equal(serializer1.args, serializer2.args)
    assert (serializer1.gate_type == serializer2.gate_type
            and serializer1.serialized_gate_id == serializer2.serialized_gate_id
            and serializer1.can_serialize_predicate == serializer2.can_serialize_predicate)


def assert_deserializers_equal(deserializer1, deserializer2):
    """Simple value equality check for op_deserializer """
    assert (
        deserializer1.serialized_gate_id == deserializer2.serialized_gate_id and
        deserializer1.gate_constructor == deserializer2.gate_constructor and
        deserializer1.args == deserializer2.args and
        deserializer1.num_qubits_param == deserializer2.num_qubits_param)


def test_serialized_gate_map():
    assert_serializers_equal(cgc.GATE_SERIALIZER[
                                 cirq.PhasedXPowGate],
                             cg.op_serializer.GateOpSerializer(
                                 gate_type=cirq.PhasedXPowGate,
                                 serialized_gate_id='exp_w',
                                 args=[
                                     cg.op_serializer.SerializingArg(
                                         serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter='phase_exponent'),
                                     cg.op_serializer.SerializingArg(
                                         serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent')
                                 ]))
    assert_serializers_equal(cgc.GATE_SERIALIZER[cirq.ZPowGate],
                             cg.op_serializer.GateOpSerializer(
                                 gate_type=cirq.ZPowGate,
                                 serialized_gate_id='exp_z',
                                 args=[
                                     cg.op_serializer.SerializingArg(
                                         serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent')
                                 ]))
    assert_serializers_equal(cgc.GATE_SERIALIZER[cirq.XPowGate],
                             cg.op_serializer.GateOpSerializer(
                                 gate_type=cirq.XPowGate,
                                 serialized_gate_id='exp_w',
                                 args=[
                                     cg.op_serializer.SerializingArg(
                                         serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter=lambda x: 0.0),
                                     cg.op_serializer.SerializingArg(
                                         serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent'),
                                 ]))
    assert_serializers_equal(cgc.GATE_SERIALIZER[cirq.YPowGate],
                             cg.op_serializer.GateOpSerializer(
                                 gate_type=cirq.YPowGate,
                                 serialized_gate_id='exp_w',
                                 args=[
                                     cg.op_serializer.SerializingArg(
                                         serialized_name='axis_half_turns',
                                         serialized_type=float,
                                         gate_getter=lambda x: 0.5),
                                     cg.op_serializer.SerializingArg(
                                         serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent'),
                                 ]))
    assert_serializers_equal(cgc.GATE_SERIALIZER[cirq.CZPowGate],
                             cg.op_serializer.GateOpSerializer(
                                 gate_type=cirq.CZPowGate,
                                 serialized_gate_id='exp_11',
                                 args=[
                                     cg.op_serializer.SerializingArg(
                                         serialized_name='half_turns',
                                         serialized_type=float,
                                         gate_getter='exponent')
                                 ]))


def test_deserialized_gate_map():
    assert_deserializers_equal(cgc.GATE_DESERIALIZER[
                                   cirq.PhasedXPowGate],
                               cg.op_deserializer.GateOpDeserializer(
                                   serialized_gate_id='exp_w',
                                   gate_constructor=cirq.PhasedXPowGate,
                                   args=[
                                       cg.op_deserializer.DeserializingArg(
                                           serialized_name='axis_half_turns',
                                           constructor_arg_name='phase_exponent'),
                                       cg.op_deserializer.DeserializingArg(
                                           serialized_name='half_turns',
                                           constructor_arg_name='exponent')
                                   ]))
    assert_deserializers_equal(cgc.GATE_DESERIALIZER[
                                   cirq.ZPowGate],
                               cg.op_deserializer.GateOpDeserializer(
                                   serialized_gate_id='exp_z',
                                   gate_constructor=cirq.ZPowGate,
                                   args=[
                                       cg.op_deserializer.DeserializingArg(
                                           serialized_name='half_turns',
                                           constructor_arg_name='exponent')
                                   ]))
    assert_deserializers_equal(cgc.GATE_DESERIALIZER[
                                   cirq.CZPowGate],
                               cg.op_deserializer.GateOpDeserializer(
                                   serialized_gate_id='exp_11',
                                   gate_constructor=cirq.CZPowGate,
                                   args=[
                                       cg.op_deserializer.DeserializingArg(
                                           serialized_name='half_turns',
                                           constructor_arg_name='exponent')
                                   ]))
