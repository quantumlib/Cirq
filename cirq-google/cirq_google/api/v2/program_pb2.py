# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cirq_google/api/v2/program.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tunits.proto import tunits_pb2 as tunits_dot_proto_dot_tunits__pb2
from . import ndarrays_pb2 as cirq__google_dot_api_dot_v2_dot_ndarrays__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n cirq_google/api/v2/program.proto\x12\x12\x63irq.google.api.v2\x1a\x19tunits/proto/tunits.proto\x1a!cirq_google/api/v2/ndarrays.proto\"\xaf\x01\n\x07Program\x12\x32\n\x08language\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.LanguageB\x02\x18\x01\x12.\n\x07\x63ircuit\x18\x02 \x01(\x0b\x32\x1b.cirq.google.api.v2.CircuitH\x00\x12/\n\tconstants\x18\x04 \x03(\x0b\x32\x1c.cirq.google.api.v2.ConstantB\t\n\x07programJ\x04\x08\x03\x10\x04\"\xaf\x02\n\x08\x43onstant\x12\x16\n\x0cstring_value\x18\x01 \x01(\tH\x00\x12\x34\n\rcircuit_value\x18\x02 \x01(\x0b\x32\x1b.cirq.google.api.v2.CircuitH\x00\x12*\n\x05qubit\x18\x03 \x01(\x0b\x32\x19.cirq.google.api.v2.QubitH\x00\x12\x32\n\x0cmoment_value\x18\x04 \x01(\x0b\x32\x1a.cirq.google.api.v2.MomentH\x00\x12\x38\n\x0foperation_value\x18\x05 \x01(\x0b\x32\x1d.cirq.google.api.v2.OperationH\x00\x12,\n\ttag_value\x18\x06 \x01(\x0b\x32\x17.cirq.google.api.v2.TagH\x00\x42\r\n\x0b\x63onst_value\"\xec\x01\n\x07\x43ircuit\x12K\n\x13scheduling_strategy\x18\x01 \x01(\x0e\x32..cirq.google.api.v2.Circuit.SchedulingStrategy\x12+\n\x07moments\x18\x02 \x03(\x0b\x32\x1a.cirq.google.api.v2.Moment\x12\x16\n\x0emoment_indices\x18\x03 \x03(\x05\"O\n\x12SchedulingStrategy\x12#\n\x1fSCHEDULING_STRATEGY_UNSPECIFIED\x10\x00\x12\x14\n\x10MOMENT_BY_MOMENT\x10\x01\"\x9e\x01\n\x06Moment\x12\x31\n\noperations\x18\x01 \x03(\x0b\x32\x1d.cirq.google.api.v2.Operation\x12@\n\x12\x63ircuit_operations\x18\x02 \x03(\x0b\x32$.cirq.google.api.v2.CircuitOperation\x12\x19\n\x11operation_indices\x18\x04 \x03(\x05J\x04\x08\x03\x10\x04\"C\n\x08Language\x12\x14\n\x08gate_set\x18\x01 \x01(\tB\x02\x18\x01\x12!\n\x15\x61rg_function_language\x18\x02 \x01(\tB\x02\x18\x01\"k\n\x08\x46loatArg\x12\x15\n\x0b\x66loat_value\x18\x01 \x01(\x02H\x00\x12\x10\n\x06symbol\x18\x02 \x01(\tH\x00\x12/\n\x04\x66unc\x18\x03 \x01(\x0b\x32\x1f.cirq.google.api.v2.ArgFunctionH\x00\x42\x05\n\x03\x61rg\":\n\x08XPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\":\n\x08YPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"Q\n\x08ZPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12\x15\n\ris_physical_z\x18\x02 \x01(\x08\"v\n\x0ePhasedXPowGate\x12\x34\n\x0ephase_exponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12.\n\x08\x65xponent\x18\x02 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"\xad\x01\n\x0cPhasedXZGate\x12\x30\n\nx_exponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12\x30\n\nz_exponent\x18\x02 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12\x39\n\x13\x61xis_phase_exponent\x18\x03 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\";\n\tCZPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"\x7f\n\x08\x46SimGate\x12+\n\x05theta\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12)\n\x03phi\x18\x02 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12\x1b\n\x13translate_via_model\x18\x03 \x01(\x08\">\n\x0cISwapPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"e\n\x0fMeasurementGate\x12$\n\x03key\x18\x01 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg\x12,\n\x0binvert_mask\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg\"@\n\x08WaitGate\x12\x34\n\x0e\x64uration_nanos\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"\xab\t\n\tOperation\x12\x30\n\x08xpowgate\x18\x07 \x01(\x0b\x32\x1c.cirq.google.api.v2.XPowGateH\x00\x12\x30\n\x08ypowgate\x18\x08 \x01(\x0b\x32\x1c.cirq.google.api.v2.YPowGateH\x00\x12\x30\n\x08zpowgate\x18\t \x01(\x0b\x32\x1c.cirq.google.api.v2.ZPowGateH\x00\x12<\n\x0ephasedxpowgate\x18\n \x01(\x0b\x32\".cirq.google.api.v2.PhasedXPowGateH\x00\x12\x38\n\x0cphasedxzgate\x18\x0b \x01(\x0b\x32 .cirq.google.api.v2.PhasedXZGateH\x00\x12\x32\n\tczpowgate\x18\x0c \x01(\x0b\x32\x1d.cirq.google.api.v2.CZPowGateH\x00\x12\x30\n\x08\x66simgate\x18\r \x01(\x0b\x32\x1c.cirq.google.api.v2.FSimGateH\x00\x12\x38\n\x0ciswappowgate\x18\x0e \x01(\x0b\x32 .cirq.google.api.v2.ISwapPowGateH\x00\x12>\n\x0fmeasurementgate\x18\x0f \x01(\x0b\x32#.cirq.google.api.v2.MeasurementGateH\x00\x12\x30\n\x08waitgate\x18\x10 \x01(\x0b\x32\x1c.cirq.google.api.v2.WaitGateH\x00\x12\x38\n\x0cinternalgate\x18\x11 \x01(\x0b\x32 .cirq.google.api.v2.InternalGateH\x00\x12@\n\x10\x63ouplerpulsegate\x18\x12 \x01(\x0b\x32$.cirq.google.api.v2.CouplerPulseGateH\x00\x12\x38\n\x0cidentitygate\x18\x13 \x01(\x0b\x32 .cirq.google.api.v2.IdentityGateH\x00\x12\x30\n\x08hpowgate\x18\x14 \x01(\x0b\x32\x1c.cirq.google.api.v2.HPowGateH\x00\x12N\n\x17singlequbitcliffordgate\x18\x15 \x01(\x0b\x32+.cirq.google.api.v2.SingleQubitCliffordGateH\x00\x12\x32\n\tresetgate\x18\x18 \x01(\x0b\x32\x1d.cirq.google.api.v2.ResetGateH\x00\x12-\n\x06qubits\x18\x03 \x03(\x0b\x32\x19.cirq.google.api.v2.QubitB\x02\x18\x01\x12\x1c\n\x14qubit_constant_index\x18\x06 \x03(\x05\x12\x15\n\x0btoken_value\x18\x04 \x01(\tH\x01\x12\x1e\n\x14token_constant_index\x18\x05 \x01(\x05H\x01\x12%\n\x04tags\x18\x16 \x03(\x0b\x32\x17.cirq.google.api.v2.Tag\x12\x13\n\x0btag_indices\x18\x17 \x03(\x05\x12/\n\x0e\x63onditioned_on\x18\x19 \x03(\x0b\x32\x17.cirq.google.api.v2.ArgB\x0c\n\ngate_valueB\x07\n\x05tokenJ\x04\x08\x01\x10\x02J\x04\x08\x02\x10\x03\"<\n\x16\x44ynamicalDecouplingTag\x12\x15\n\x08protocol\x18\x01 \x01(\tH\x00\x88\x01\x01\x42\x0b\n\t_protocol\"\xb6\x03\n\x03Tag\x12J\n\x14\x64ynamical_decoupling\x18\x01 \x01(\x0b\x32*.cirq.google.api.v2.DynamicalDecouplingTagH\x00\x12\x30\n\x07no_sync\x18\x02 \x01(\x0b\x32\x1d.cirq.google.api.v2.NoSyncTagH\x00\x12\x38\n\x0bphase_match\x18\x03 \x01(\x0b\x32!.cirq.google.api.v2.PhaseMatchTagH\x00\x12\x36\n\nphysical_z\x18\x04 \x01(\x0b\x32 .cirq.google.api.v2.PhysicalZTagH\x00\x12@\n\x0f\x63lassical_state\x18\x05 \x01(\x0b\x32%.cirq.google.api.v2.ClassicalStateTagH\x00\x12=\n\x0e\x66sim_via_model\x18\x07 \x01(\x0b\x32#.cirq.google.api.v2.FSimViaModelTagH\x00\x12\x37\n\x0cinternal_tag\x18\x08 \x01(\x0b\x32\x1f.cirq.google.api.v2.InternalTagH\x00\x42\x05\n\x03tag\"\x0f\n\rPhaseMatchTag\"\x0e\n\x0cPhysicalZTag\"\x13\n\x11\x43lassicalStateTag\"\x11\n\x0f\x46SimViaModelTag\"\x84\x01\n\tNoSyncTag\x12\x11\n\x07reverse\x18\x01 \x01(\x05H\x00\x12!\n\x17remove_all_syncs_before\x18\x02 \x01(\x08H\x00\x12\x11\n\x07\x66orward\x18\x03 \x01(\x05H\x01\x12 \n\x16remove_all_syncs_after\x18\x04 \x01(\x08H\x01\x42\x05\n\x03revB\x05\n\x03\x66wd\"\xd5\x02\n\x0bInternalTag\x12\x10\n\x08tag_name\x18\x01 \x01(\t\x12\x13\n\x0btag_package\x18\x02 \x01(\t\x12>\n\x08tag_args\x18\x03 \x03(\x0b\x32,.cirq.google.api.v2.InternalTag.TagArgsEntry\x12\x44\n\x0b\x63ustom_args\x18\x04 \x03(\x0b\x32/.cirq.google.api.v2.InternalTag.CustomArgsEntry\x1aG\n\x0cTagArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg:\x02\x38\x01\x1aP\n\x0f\x43ustomArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12,\n\x05value\x18\x02 \x01(\x0b\x32\x1d.cirq.google.api.v2.CustomArg:\x02\x38\x01\"\x12\n\x04Gate\x12\n\n\x02id\x18\x01 \x01(\t\"\x13\n\x05Qubit\x12\n\n\x02id\x18\x02 \x01(\t\"\xdb\x01\n\x03\x41rg\x12\x31\n\targ_value\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.ArgValueH\x00\x12\x10\n\x06symbol\x18\x02 \x01(\tH\x00\x12/\n\x04\x66unc\x18\x03 \x01(\x0b\x32\x1f.cirq.google.api.v2.ArgFunctionH\x00\x12\x18\n\x0e\x63onstant_index\x18\x04 \x01(\x05H\x00\x12=\n\x0fmeasurement_key\x18\x05 \x01(\x0b\x32\".cirq.google.api.v2.MeasurementKeyH\x00\x42\x05\n\x03\x61rg\"\xc4\x04\n\x08\x41rgValue\x12\x15\n\x0b\x66loat_value\x18\x01 \x01(\x02H\x00\x12:\n\x0b\x62ool_values\x18\x02 \x01(\x0b\x32#.cirq.google.api.v2.RepeatedBooleanH\x00\x12\x16\n\x0cstring_value\x18\x03 \x01(\tH\x00\x12\x16\n\x0c\x64ouble_value\x18\x04 \x01(\x01H\x00\x12\x39\n\x0cint64_values\x18\x05 \x01(\x0b\x32!.cirq.google.api.v2.RepeatedInt64H\x00\x12;\n\rdouble_values\x18\x06 \x01(\x0b\x32\".cirq.google.api.v2.RepeatedDoubleH\x00\x12;\n\rstring_values\x18\x07 \x01(\x0b\x32\".cirq.google.api.v2.RepeatedStringH\x00\x12(\n\x0fvalue_with_unit\x18\x08 \x01(\x0b\x32\r.tunits.ValueH\x00\x12\x14\n\nbool_value\x18\t \x01(\x08H\x00\x12\x15\n\x0b\x62ytes_value\x18\n \x01(\x0cH\x00\x12\x34\n\rcomplex_value\x18\x0b \x01(\x0b\x32\x1b.cirq.google.api.v2.ComplexH\x00\x12\x30\n\x0btuple_value\x18\x0c \x01(\x0b\x32\x19.cirq.google.api.v2.TupleH\x00\x12\x34\n\rndarray_value\x18\r \x01(\x0b\x32\x1b.cirq.google.api.v2.NDArrayH\x00\x42\x0b\n\targ_value\"\x1f\n\rRepeatedInt64\x12\x0e\n\x06values\x18\x01 \x03(\x03\" \n\x0eRepeatedDouble\x12\x0e\n\x06values\x18\x01 \x03(\x01\" \n\x0eRepeatedString\x12\x0e\n\x06values\x18\x01 \x03(\t\"!\n\x0fRepeatedBoolean\x12\x0e\n\x06values\x18\x01 \x03(\x08\"\xbd\x01\n\x05Tuple\x12=\n\rsequence_type\x18\x01 \x01(\x0e\x32&.cirq.google.api.v2.Tuple.SequenceType\x12\'\n\x06values\x18\x02 \x03(\x0b\x32\x17.cirq.google.api.v2.Arg\"L\n\x0cSequenceType\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x08\n\x04LIST\x10\x01\x12\t\n\x05TUPLE\x10\x02\x12\x07\n\x03SET\x10\x03\x12\r\n\tFROZENSET\x10\x04\"1\n\x07\x43omplex\x12\x12\n\nreal_value\x18\x01 \x01(\x01\x12\x12\n\nimag_value\x18\x02 \x01(\x01\"\x85\x05\n\x07NDArray\x12?\n\x10\x63omplex128_array\x18\x01 \x01(\x0b\x32#.cirq.google.api.v2.Complex128ArrayH\x00\x12=\n\x0f\x63omplex64_array\x18\x02 \x01(\x0b\x32\".cirq.google.api.v2.Complex64ArrayH\x00\x12\x39\n\rfloat16_array\x18\x03 \x01(\x0b\x32 .cirq.google.api.v2.Float16ArrayH\x00\x12\x39\n\rfloat32_array\x18\x04 \x01(\x0b\x32 .cirq.google.api.v2.Float32ArrayH\x00\x12\x39\n\rfloat64_array\x18\x05 \x01(\x0b\x32 .cirq.google.api.v2.Float64ArrayH\x00\x12\x35\n\x0bint64_array\x18\x06 \x01(\x0b\x32\x1e.cirq.google.api.v2.Int64ArrayH\x00\x12\x35\n\x0bint32_array\x18\x07 \x01(\x0b\x32\x1e.cirq.google.api.v2.Int32ArrayH\x00\x12\x35\n\x0bint16_array\x18\x08 \x01(\x0b\x32\x1e.cirq.google.api.v2.Int16ArrayH\x00\x12\x33\n\nint8_array\x18\t \x01(\x0b\x32\x1d.cirq.google.api.v2.Int8ArrayH\x00\x12\x35\n\x0buint8_array\x18\n \x01(\x0b\x32\x1e.cirq.google.api.v2.UInt8ArrayH\x00\x12\x31\n\tbit_array\x18\x0b \x01(\x0b\x32\x1c.cirq.google.api.v2.BitArrayH\x00\x42\x05\n\x03\x61rr\"B\n\x0b\x41rgFunction\x12\x0c\n\x04type\x18\x01 \x01(\t\x12%\n\x04\x61rgs\x18\x02 \x03(\x0b\x32\x17.cirq.google.api.v2.Arg\"\xa5\x03\n\x10\x43ircuitOperation\x12\x1e\n\x16\x63ircuit_constant_index\x18\x01 \x01(\x05\x12M\n\x18repetition_specification\x18\x02 \x01(\x0b\x32+.cirq.google.api.v2.RepetitionSpecification\x12\x33\n\tqubit_map\x18\x03 \x01(\x0b\x32 .cirq.google.api.v2.QubitMapping\x12\x46\n\x13measurement_key_map\x18\x04 \x01(\x0b\x32).cirq.google.api.v2.MeasurementKeyMapping\x12/\n\x07\x61rg_map\x18\x05 \x01(\x0b\x32\x1e.cirq.google.api.v2.ArgMapping\x12\x32\n\x0crepeat_until\x18\x06 \x01(\x0b\x32\x17.cirq.google.api.v2.ArgH\x00\x88\x01\x01\x12/\n\x0e\x63onditioned_on\x18\x07 \x03(\x0b\x32\x17.cirq.google.api.v2.ArgB\x0f\n\r_repeat_until\"\xbc\x01\n\x17RepetitionSpecification\x12S\n\x0erepetition_ids\x18\x01 \x01(\x0b\x32\x39.cirq.google.api.v2.RepetitionSpecification.RepetitionIdsH\x00\x12\x1a\n\x10repetition_count\x18\x02 \x01(\x05H\x00\x1a\x1c\n\rRepetitionIds\x12\x0b\n\x03ids\x18\x01 \x03(\tB\x12\n\x10repetition_value\"\xac\x01\n\x0cQubitMapping\x12<\n\x07\x65ntries\x18\x01 \x03(\x0b\x32+.cirq.google.api.v2.QubitMapping.QubitEntry\x1a^\n\nQubitEntry\x12&\n\x03key\x18\x01 \x01(\x0b\x32\x19.cirq.google.api.v2.Qubit\x12(\n\x05value\x18\x02 \x01(\x0b\x32\x19.cirq.google.api.v2.Qubit\"P\n\x0eMeasurementKey\x12\x12\n\nstring_key\x18\x01 \x01(\t\x12\x0c\n\x04path\x18\x02 \x03(\t\x12\x12\n\x05index\x18\x03 \x01(\x05H\x00\x88\x01\x01\x42\x08\n\x06_index\"\xe2\x01\n\x15MeasurementKeyMapping\x12N\n\x07\x65ntries\x18\x01 \x03(\x0b\x32=.cirq.google.api.v2.MeasurementKeyMapping.MeasurementKeyEntry\x1ay\n\x13MeasurementKeyEntry\x12/\n\x03key\x18\x01 \x01(\x0b\x32\".cirq.google.api.v2.MeasurementKey\x12\x31\n\x05value\x18\x02 \x01(\x0b\x32\".cirq.google.api.v2.MeasurementKey\"\xa0\x01\n\nArgMapping\x12\x38\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\'.cirq.google.api.v2.ArgMapping.ArgEntry\x1aX\n\x08\x41rgEntry\x12$\n\x03key\x18\x01 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg\"C\n\x15\x46unctionInterpolation\x12\x14\n\x08x_values\x18\x01 \x03(\x02\x42\x02\x10\x01\x12\x14\n\x08y_values\x18\x02 \x03(\x02\x42\x02\x10\x01\"k\n\tCustomArg\x12P\n\x1b\x66unction_interpolation_data\x18\x01 \x01(\x0b\x32).cirq.google.api.v2.FunctionInterpolationH\x00\x42\x0c\n\ncustom_arg\"\xe6\x02\n\x0cInternalGate\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06module\x18\x02 \x01(\t\x12\x12\n\nnum_qubits\x18\x03 \x01(\x05\x12\x41\n\tgate_args\x18\x04 \x03(\x0b\x32..cirq.google.api.v2.InternalGate.GateArgsEntry\x12\x45\n\x0b\x63ustom_args\x18\x05 \x03(\x0b\x32\x30.cirq.google.api.v2.InternalGate.CustomArgsEntry\x1aH\n\rGateArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg:\x02\x38\x01\x1aP\n\x0f\x43ustomArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12,\n\x05value\x18\x02 \x01(\x0b\x32\x1d.cirq.google.api.v2.CustomArg:\x02\x38\x01\"\xd8\x03\n\x10\x43ouplerPulseGate\x12\x37\n\x0chold_time_ps\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x00\x88\x01\x01\x12\x37\n\x0crise_time_ps\x18\x02 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x01\x88\x01\x01\x12:\n\x0fpadding_time_ps\x18\x03 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x02\x88\x01\x01\x12\x37\n\x0c\x63oupling_mhz\x18\x04 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x03\x88\x01\x01\x12\x38\n\rq0_detune_mhz\x18\x05 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x04\x88\x01\x01\x12\x38\n\rq1_detune_mhz\x18\x06 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x05\x88\x01\x01\x42\x0f\n\r_hold_time_psB\x0f\n\r_rise_time_psB\x12\n\x10_padding_time_psB\x0f\n\r_coupling_mhzB\x10\n\x0e_q0_detune_mhzB\x10\n\x0e_q1_detune_mhz\"\x8b\x01\n\x0f\x43liffordTableau\x12\x17\n\nnum_qubits\x18\x01 \x01(\x05H\x00\x88\x01\x01\x12\x1a\n\rinitial_state\x18\x02 \x01(\x05H\x01\x88\x01\x01\x12\n\n\x02rs\x18\x03 \x03(\x08\x12\n\n\x02xs\x18\x04 \x03(\x08\x12\n\n\x02zs\x18\x05 \x03(\x08\x42\r\n\x0b_num_qubitsB\x10\n\x0e_initial_state\"O\n\x17SingleQubitCliffordGate\x12\x34\n\x07tableau\x18\x01 \x01(\x0b\x32#.cirq.google.api.v2.CliffordTableau\"!\n\x0cIdentityGate\x12\x11\n\tqid_shape\x18\x01 \x03(\r\":\n\x08HPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"\xab\x01\n\tResetGate\x12\x12\n\nreset_type\x18\x01 \x01(\t\x12?\n\targuments\x18\x02 \x03(\x0b\x32,.cirq.google.api.v2.ResetGate.ArgumentsEntry\x1aI\n\x0e\x41rgumentsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg:\x02\x38\x01\x42/\n\x1d\x63om.google.cirq.google.api.v2B\x0cProgramProtoP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'cirq_google.api.v2.program_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\035com.google.cirq.google.api.v2B\014ProgramProtoP\001'
  _globals['_PROGRAM'].fields_by_name['language']._options = None
  _globals['_PROGRAM'].fields_by_name['language']._serialized_options = b'\030\001'
  _globals['_LANGUAGE'].fields_by_name['gate_set']._options = None
  _globals['_LANGUAGE'].fields_by_name['gate_set']._serialized_options = b'\030\001'
  _globals['_LANGUAGE'].fields_by_name['arg_function_language']._options = None
  _globals['_LANGUAGE'].fields_by_name['arg_function_language']._serialized_options = b'\030\001'
  _globals['_OPERATION'].fields_by_name['qubits']._options = None
  _globals['_OPERATION'].fields_by_name['qubits']._serialized_options = b'\030\001'
  _globals['_INTERNALTAG_TAGARGSENTRY']._options = None
  _globals['_INTERNALTAG_TAGARGSENTRY']._serialized_options = b'8\001'
  _globals['_INTERNALTAG_CUSTOMARGSENTRY']._options = None
  _globals['_INTERNALTAG_CUSTOMARGSENTRY']._serialized_options = b'8\001'
  _globals['_FUNCTIONINTERPOLATION'].fields_by_name['x_values']._options = None
  _globals['_FUNCTIONINTERPOLATION'].fields_by_name['x_values']._serialized_options = b'\020\001'
  _globals['_FUNCTIONINTERPOLATION'].fields_by_name['y_values']._options = None
  _globals['_FUNCTIONINTERPOLATION'].fields_by_name['y_values']._serialized_options = b'\020\001'
  _globals['_INTERNALGATE_GATEARGSENTRY']._options = None
  _globals['_INTERNALGATE_GATEARGSENTRY']._serialized_options = b'8\001'
  _globals['_INTERNALGATE_CUSTOMARGSENTRY']._options = None
  _globals['_INTERNALGATE_CUSTOMARGSENTRY']._serialized_options = b'8\001'
  _globals['_RESETGATE_ARGUMENTSENTRY']._options = None
  _globals['_RESETGATE_ARGUMENTSENTRY']._serialized_options = b'8\001'
  _globals['_PROGRAM']._serialized_start=119
  _globals['_PROGRAM']._serialized_end=294
  _globals['_CONSTANT']._serialized_start=297
  _globals['_CONSTANT']._serialized_end=600
  _globals['_CIRCUIT']._serialized_start=603
  _globals['_CIRCUIT']._serialized_end=839
  _globals['_CIRCUIT_SCHEDULINGSTRATEGY']._serialized_start=760
  _globals['_CIRCUIT_SCHEDULINGSTRATEGY']._serialized_end=839
  _globals['_MOMENT']._serialized_start=842
  _globals['_MOMENT']._serialized_end=1000
  _globals['_LANGUAGE']._serialized_start=1002
  _globals['_LANGUAGE']._serialized_end=1069
  _globals['_FLOATARG']._serialized_start=1071
  _globals['_FLOATARG']._serialized_end=1178
  _globals['_XPOWGATE']._serialized_start=1180
  _globals['_XPOWGATE']._serialized_end=1238
  _globals['_YPOWGATE']._serialized_start=1240
  _globals['_YPOWGATE']._serialized_end=1298
  _globals['_ZPOWGATE']._serialized_start=1300
  _globals['_ZPOWGATE']._serialized_end=1381
  _globals['_PHASEDXPOWGATE']._serialized_start=1383
  _globals['_PHASEDXPOWGATE']._serialized_end=1501
  _globals['_PHASEDXZGATE']._serialized_start=1504
  _globals['_PHASEDXZGATE']._serialized_end=1677
  _globals['_CZPOWGATE']._serialized_start=1679
  _globals['_CZPOWGATE']._serialized_end=1738
  _globals['_FSIMGATE']._serialized_start=1740
  _globals['_FSIMGATE']._serialized_end=1867
  _globals['_ISWAPPOWGATE']._serialized_start=1869
  _globals['_ISWAPPOWGATE']._serialized_end=1931
  _globals['_MEASUREMENTGATE']._serialized_start=1933
  _globals['_MEASUREMENTGATE']._serialized_end=2034
  _globals['_WAITGATE']._serialized_start=2036
  _globals['_WAITGATE']._serialized_end=2100
  _globals['_OPERATION']._serialized_start=2103
  _globals['_OPERATION']._serialized_end=3298
  _globals['_DYNAMICALDECOUPLINGTAG']._serialized_start=3300
  _globals['_DYNAMICALDECOUPLINGTAG']._serialized_end=3360
  _globals['_TAG']._serialized_start=3363
  _globals['_TAG']._serialized_end=3801
  _globals['_PHASEMATCHTAG']._serialized_start=3803
  _globals['_PHASEMATCHTAG']._serialized_end=3818
  _globals['_PHYSICALZTAG']._serialized_start=3820
  _globals['_PHYSICALZTAG']._serialized_end=3834
  _globals['_CLASSICALSTATETAG']._serialized_start=3836
  _globals['_CLASSICALSTATETAG']._serialized_end=3855
  _globals['_FSIMVIAMODELTAG']._serialized_start=3857
  _globals['_FSIMVIAMODELTAG']._serialized_end=3874
  _globals['_NOSYNCTAG']._serialized_start=3877
  _globals['_NOSYNCTAG']._serialized_end=4009
  _globals['_INTERNALTAG']._serialized_start=4012
  _globals['_INTERNALTAG']._serialized_end=4353
  _globals['_INTERNALTAG_TAGARGSENTRY']._serialized_start=4200
  _globals['_INTERNALTAG_TAGARGSENTRY']._serialized_end=4271
  _globals['_INTERNALTAG_CUSTOMARGSENTRY']._serialized_start=4273
  _globals['_INTERNALTAG_CUSTOMARGSENTRY']._serialized_end=4353
  _globals['_GATE']._serialized_start=4355
  _globals['_GATE']._serialized_end=4373
  _globals['_QUBIT']._serialized_start=4375
  _globals['_QUBIT']._serialized_end=4394
  _globals['_ARG']._serialized_start=4397
  _globals['_ARG']._serialized_end=4616
  _globals['_ARGVALUE']._serialized_start=4619
  _globals['_ARGVALUE']._serialized_end=5199
  _globals['_REPEATEDINT64']._serialized_start=5201
  _globals['_REPEATEDINT64']._serialized_end=5232
  _globals['_REPEATEDDOUBLE']._serialized_start=5234
  _globals['_REPEATEDDOUBLE']._serialized_end=5266
  _globals['_REPEATEDSTRING']._serialized_start=5268
  _globals['_REPEATEDSTRING']._serialized_end=5300
  _globals['_REPEATEDBOOLEAN']._serialized_start=5302
  _globals['_REPEATEDBOOLEAN']._serialized_end=5335
  _globals['_TUPLE']._serialized_start=5338
  _globals['_TUPLE']._serialized_end=5527
  _globals['_TUPLE_SEQUENCETYPE']._serialized_start=5451
  _globals['_TUPLE_SEQUENCETYPE']._serialized_end=5527
  _globals['_COMPLEX']._serialized_start=5529
  _globals['_COMPLEX']._serialized_end=5578
  _globals['_NDARRAY']._serialized_start=5581
  _globals['_NDARRAY']._serialized_end=6226
  _globals['_ARGFUNCTION']._serialized_start=6228
  _globals['_ARGFUNCTION']._serialized_end=6294
  _globals['_CIRCUITOPERATION']._serialized_start=6297
  _globals['_CIRCUITOPERATION']._serialized_end=6718
  _globals['_REPETITIONSPECIFICATION']._serialized_start=6721
  _globals['_REPETITIONSPECIFICATION']._serialized_end=6909
  _globals['_REPETITIONSPECIFICATION_REPETITIONIDS']._serialized_start=6861
  _globals['_REPETITIONSPECIFICATION_REPETITIONIDS']._serialized_end=6889
  _globals['_QUBITMAPPING']._serialized_start=6912
  _globals['_QUBITMAPPING']._serialized_end=7084
  _globals['_QUBITMAPPING_QUBITENTRY']._serialized_start=6990
  _globals['_QUBITMAPPING_QUBITENTRY']._serialized_end=7084
  _globals['_MEASUREMENTKEY']._serialized_start=7086
  _globals['_MEASUREMENTKEY']._serialized_end=7166
  _globals['_MEASUREMENTKEYMAPPING']._serialized_start=7169
  _globals['_MEASUREMENTKEYMAPPING']._serialized_end=7395
  _globals['_MEASUREMENTKEYMAPPING_MEASUREMENTKEYENTRY']._serialized_start=7274
  _globals['_MEASUREMENTKEYMAPPING_MEASUREMENTKEYENTRY']._serialized_end=7395
  _globals['_ARGMAPPING']._serialized_start=7398
  _globals['_ARGMAPPING']._serialized_end=7558
  _globals['_ARGMAPPING_ARGENTRY']._serialized_start=7470
  _globals['_ARGMAPPING_ARGENTRY']._serialized_end=7558
  _globals['_FUNCTIONINTERPOLATION']._serialized_start=7560
  _globals['_FUNCTIONINTERPOLATION']._serialized_end=7627
  _globals['_CUSTOMARG']._serialized_start=7629
  _globals['_CUSTOMARG']._serialized_end=7736
  _globals['_INTERNALGATE']._serialized_start=7739
  _globals['_INTERNALGATE']._serialized_end=8097
  _globals['_INTERNALGATE_GATEARGSENTRY']._serialized_start=7943
  _globals['_INTERNALGATE_GATEARGSENTRY']._serialized_end=8015
  _globals['_INTERNALGATE_CUSTOMARGSENTRY']._serialized_start=4273
  _globals['_INTERNALGATE_CUSTOMARGSENTRY']._serialized_end=4353
  _globals['_COUPLERPULSEGATE']._serialized_start=8100
  _globals['_COUPLERPULSEGATE']._serialized_end=8572
  _globals['_CLIFFORDTABLEAU']._serialized_start=8575
  _globals['_CLIFFORDTABLEAU']._serialized_end=8714
  _globals['_SINGLEQUBITCLIFFORDGATE']._serialized_start=8716
  _globals['_SINGLEQUBITCLIFFORDGATE']._serialized_end=8795
  _globals['_IDENTITYGATE']._serialized_start=8797
  _globals['_IDENTITYGATE']._serialized_end=8830
  _globals['_HPOWGATE']._serialized_start=8832
  _globals['_HPOWGATE']._serialized_end=8890
  _globals['_RESETGATE']._serialized_start=8893
  _globals['_RESETGATE']._serialized_end=9064
  _globals['_RESETGATE_ARGUMENTSENTRY']._serialized_start=8991
  _globals['_RESETGATE_ARGUMENTSENTRY']._serialized_end=9064
# @@protoc_insertion_point(module_scope)
