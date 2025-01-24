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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n cirq_google/api/v2/program.proto\x12\x12\x63irq.google.api.v2\x1a\x19tunits/proto/tunits.proto\"\xd7\x01\n\x07Program\x12.\n\x08language\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.Language\x12.\n\x07\x63ircuit\x18\x02 \x01(\x0b\x32\x1b.cirq.google.api.v2.CircuitH\x00\x12\x30\n\x08schedule\x18\x03 \x01(\x0b\x32\x1c.cirq.google.api.v2.ScheduleH\x00\x12/\n\tconstants\x18\x04 \x03(\x0b\x32\x1c.cirq.google.api.v2.ConstantB\t\n\x07program\"\xc7\x01\n\x08\x43onstant\x12\x16\n\x0cstring_value\x18\x01 \x01(\tH\x00\x12\x34\n\rcircuit_value\x18\x02 \x01(\x0b\x32\x1b.cirq.google.api.v2.CircuitH\x00\x12*\n\x05qubit\x18\x03 \x01(\x0b\x32\x19.cirq.google.api.v2.QubitH\x00\x12\x32\n\x0cmoment_value\x18\x04 \x01(\x0b\x32\x1a.cirq.google.api.v2.MomentH\x00\x42\r\n\x0b\x63onst_value\"\xd4\x01\n\x07\x43ircuit\x12K\n\x13scheduling_strategy\x18\x01 \x01(\x0e\x32..cirq.google.api.v2.Circuit.SchedulingStrategy\x12+\n\x07moments\x18\x02 \x03(\x0b\x32\x1a.cirq.google.api.v2.Moment\"O\n\x12SchedulingStrategy\x12#\n\x1fSCHEDULING_STRATEGY_UNSPECIFIED\x10\x00\x12\x14\n\x10MOMENT_BY_MOMENT\x10\x01\"\xbb\x01\n\x06Moment\x12\"\n\x15moment_constant_index\x18\x03 \x01(\x05H\x00\x88\x01\x01\x12\x31\n\noperations\x18\x01 \x03(\x0b\x32\x1d.cirq.google.api.v2.Operation\x12@\n\x12\x63ircuit_operations\x18\x02 \x03(\x0b\x32$.cirq.google.api.v2.CircuitOperationB\x18\n\x16_moment_constant_index\"P\n\x08Schedule\x12\x44\n\x14scheduled_operations\x18\x03 \x03(\x0b\x32&.cirq.google.api.v2.ScheduledOperation\"`\n\x12ScheduledOperation\x12\x30\n\toperation\x18\x01 \x01(\x0b\x32\x1d.cirq.google.api.v2.Operation\x12\x18\n\x10start_time_picos\x18\x02 \x01(\x03\"?\n\x08Language\x12\x14\n\x08gate_set\x18\x01 \x01(\tB\x02\x18\x01\x12\x1d\n\x15\x61rg_function_language\x18\x02 \x01(\t\"k\n\x08\x46loatArg\x12\x15\n\x0b\x66loat_value\x18\x01 \x01(\x02H\x00\x12\x10\n\x06symbol\x18\x02 \x01(\tH\x00\x12/\n\x04\x66unc\x18\x03 \x01(\x0b\x32\x1f.cirq.google.api.v2.ArgFunctionH\x00\x42\x05\n\x03\x61rg\":\n\x08XPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\":\n\x08YPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"Q\n\x08ZPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12\x15\n\ris_physical_z\x18\x02 \x01(\x08\"v\n\x0ePhasedXPowGate\x12\x34\n\x0ephase_exponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12.\n\x08\x65xponent\x18\x02 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"\xad\x01\n\x0cPhasedXZGate\x12\x30\n\nx_exponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12\x30\n\nz_exponent\x18\x02 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12\x39\n\x13\x61xis_phase_exponent\x18\x03 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\";\n\tCZPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"\x7f\n\x08\x46SimGate\x12+\n\x05theta\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12)\n\x03phi\x18\x02 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\x12\x1b\n\x13translate_via_model\x18\x03 \x01(\x08\">\n\x0cISwapPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"e\n\x0fMeasurementGate\x12$\n\x03key\x18\x01 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg\x12,\n\x0binvert_mask\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg\"@\n\x08WaitGate\x12\x34\n\x0e\x64uration_nanos\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArg\"\xce\t\n\tOperation\x12*\n\x04gate\x18\x01 \x01(\x0b\x32\x18.cirq.google.api.v2.GateB\x02\x18\x01\x12\x30\n\x08xpowgate\x18\x07 \x01(\x0b\x32\x1c.cirq.google.api.v2.XPowGateH\x00\x12\x30\n\x08ypowgate\x18\x08 \x01(\x0b\x32\x1c.cirq.google.api.v2.YPowGateH\x00\x12\x30\n\x08zpowgate\x18\t \x01(\x0b\x32\x1c.cirq.google.api.v2.ZPowGateH\x00\x12<\n\x0ephasedxpowgate\x18\n \x01(\x0b\x32\".cirq.google.api.v2.PhasedXPowGateH\x00\x12\x38\n\x0cphasedxzgate\x18\x0b \x01(\x0b\x32 .cirq.google.api.v2.PhasedXZGateH\x00\x12\x32\n\tczpowgate\x18\x0c \x01(\x0b\x32\x1d.cirq.google.api.v2.CZPowGateH\x00\x12\x30\n\x08\x66simgate\x18\r \x01(\x0b\x32\x1c.cirq.google.api.v2.FSimGateH\x00\x12\x38\n\x0ciswappowgate\x18\x0e \x01(\x0b\x32 .cirq.google.api.v2.ISwapPowGateH\x00\x12>\n\x0fmeasurementgate\x18\x0f \x01(\x0b\x32#.cirq.google.api.v2.MeasurementGateH\x00\x12\x30\n\x08waitgate\x18\x10 \x01(\x0b\x32\x1c.cirq.google.api.v2.WaitGateH\x00\x12\x38\n\x0cinternalgate\x18\x11 \x01(\x0b\x32 .cirq.google.api.v2.InternalGateH\x00\x12@\n\x10\x63ouplerpulsegate\x18\x12 \x01(\x0b\x32$.cirq.google.api.v2.CouplerPulseGateH\x00\x12\x38\n\x0cidentitygate\x18\x13 \x01(\x0b\x32 .cirq.google.api.v2.IdentityGateH\x00\x12\x30\n\x08hpowgate\x18\x14 \x01(\x0b\x32\x1c.cirq.google.api.v2.HPowGateH\x00\x12N\n\x17singlequbitcliffordgate\x18\x15 \x01(\x0b\x32+.cirq.google.api.v2.SingleQubitCliffordGateH\x00\x12\x39\n\x04\x61rgs\x18\x02 \x03(\x0b\x32\'.cirq.google.api.v2.Operation.ArgsEntryB\x02\x18\x01\x12)\n\x06qubits\x18\x03 \x03(\x0b\x32\x19.cirq.google.api.v2.Qubit\x12\x1c\n\x14qubit_constant_index\x18\x06 \x03(\x05\x12\x15\n\x0btoken_value\x18\x04 \x01(\tH\x01\x12\x1e\n\x14token_constant_index\x18\x05 \x01(\x05H\x01\x12%\n\x04tags\x18\x16 \x03(\x0b\x32\x17.cirq.google.api.v2.Tag\x1a\x44\n\tArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg:\x02\x38\x01\x42\x0c\n\ngate_valueB\x07\n\x05token\"<\n\x16\x44ynamicalDecouplingTag\x12\x15\n\x08protocol\x18\x01 \x01(\tH\x00\x88\x01\x01\x42\x0b\n\t_protocol\"X\n\x03Tag\x12J\n\x14\x64ynamical_decoupling\x18\x01 \x01(\x0b\x32*.cirq.google.api.v2.DynamicalDecouplingTagH\x00\x42\x05\n\x03tag\"\x12\n\x04Gate\x12\n\n\x02id\x18\x01 \x01(\t\"\x13\n\x05Qubit\x12\n\n\x02id\x18\x02 \x01(\t\"\x9c\x01\n\x03\x41rg\x12\x31\n\targ_value\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.ArgValueH\x00\x12\x10\n\x06symbol\x18\x02 \x01(\tH\x00\x12/\n\x04\x66unc\x18\x03 \x01(\x0b\x32\x1f.cirq.google.api.v2.ArgFunctionH\x00\x12\x18\n\x0e\x63onstant_index\x18\x04 \x01(\x05H\x00\x42\x05\n\x03\x61rg\"\xf9\x02\n\x08\x41rgValue\x12\x15\n\x0b\x66loat_value\x18\x01 \x01(\x02H\x00\x12:\n\x0b\x62ool_values\x18\x02 \x01(\x0b\x32#.cirq.google.api.v2.RepeatedBooleanH\x00\x12\x16\n\x0cstring_value\x18\x03 \x01(\tH\x00\x12\x16\n\x0c\x64ouble_value\x18\x04 \x01(\x01H\x00\x12\x39\n\x0cint64_values\x18\x05 \x01(\x0b\x32!.cirq.google.api.v2.RepeatedInt64H\x00\x12;\n\rdouble_values\x18\x06 \x01(\x0b\x32\".cirq.google.api.v2.RepeatedDoubleH\x00\x12;\n\rstring_values\x18\x07 \x01(\x0b\x32\".cirq.google.api.v2.RepeatedStringH\x00\x12(\n\x0fvalue_with_unit\x18\x08 \x01(\x0b\x32\r.tunits.ValueH\x00\x42\x0b\n\targ_value\"\x1f\n\rRepeatedInt64\x12\x0e\n\x06values\x18\x01 \x03(\x03\" \n\x0eRepeatedDouble\x12\x0e\n\x06values\x18\x01 \x03(\x01\" \n\x0eRepeatedString\x12\x0e\n\x06values\x18\x01 \x03(\t\"!\n\x0fRepeatedBoolean\x12\x0e\n\x06values\x18\x01 \x03(\x08\"B\n\x0b\x41rgFunction\x12\x0c\n\x04type\x18\x01 \x01(\t\x12%\n\x04\x61rgs\x18\x02 \x03(\x0b\x32\x17.cirq.google.api.v2.Arg\"\xaf\x02\n\x10\x43ircuitOperation\x12\x1e\n\x16\x63ircuit_constant_index\x18\x01 \x01(\x05\x12M\n\x18repetition_specification\x18\x02 \x01(\x0b\x32+.cirq.google.api.v2.RepetitionSpecification\x12\x33\n\tqubit_map\x18\x03 \x01(\x0b\x32 .cirq.google.api.v2.QubitMapping\x12\x46\n\x13measurement_key_map\x18\x04 \x01(\x0b\x32).cirq.google.api.v2.MeasurementKeyMapping\x12/\n\x07\x61rg_map\x18\x05 \x01(\x0b\x32\x1e.cirq.google.api.v2.ArgMapping\"\xbc\x01\n\x17RepetitionSpecification\x12S\n\x0erepetition_ids\x18\x01 \x01(\x0b\x32\x39.cirq.google.api.v2.RepetitionSpecification.RepetitionIdsH\x00\x12\x1a\n\x10repetition_count\x18\x02 \x01(\x05H\x00\x1a\x1c\n\rRepetitionIds\x12\x0b\n\x03ids\x18\x01 \x03(\tB\x12\n\x10repetition_value\"\xac\x01\n\x0cQubitMapping\x12<\n\x07\x65ntries\x18\x01 \x03(\x0b\x32+.cirq.google.api.v2.QubitMapping.QubitEntry\x1a^\n\nQubitEntry\x12&\n\x03key\x18\x01 \x01(\x0b\x32\x19.cirq.google.api.v2.Qubit\x12(\n\x05value\x18\x02 \x01(\x0b\x32\x19.cirq.google.api.v2.Qubit\"$\n\x0eMeasurementKey\x12\x12\n\nstring_key\x18\x01 \x01(\t\"\xe2\x01\n\x15MeasurementKeyMapping\x12N\n\x07\x65ntries\x18\x01 \x03(\x0b\x32=.cirq.google.api.v2.MeasurementKeyMapping.MeasurementKeyEntry\x1ay\n\x13MeasurementKeyEntry\x12/\n\x03key\x18\x01 \x01(\x0b\x32\".cirq.google.api.v2.MeasurementKey\x12\x31\n\x05value\x18\x02 \x01(\x0b\x32\".cirq.google.api.v2.MeasurementKey\"\xa0\x01\n\nArgMapping\x12\x38\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\'.cirq.google.api.v2.ArgMapping.ArgEntry\x1aX\n\x08\x41rgEntry\x12$\n\x03key\x18\x01 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg\"C\n\x15\x46unctionInterpolation\x12\x14\n\x08x_values\x18\x01 \x03(\x02\x42\x02\x10\x01\x12\x14\n\x08y_values\x18\x02 \x03(\x02\x42\x02\x10\x01\"k\n\tCustomArg\x12P\n\x1b\x66unction_interpolation_data\x18\x01 \x01(\x0b\x32).cirq.google.api.v2.FunctionInterpolationH\x00\x42\x0c\n\ncustom_arg\"\xe6\x02\n\x0cInternalGate\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06module\x18\x02 \x01(\t\x12\x12\n\nnum_qubits\x18\x03 \x01(\x05\x12\x41\n\tgate_args\x18\x04 \x03(\x0b\x32..cirq.google.api.v2.InternalGate.GateArgsEntry\x12\x45\n\x0b\x63ustom_args\x18\x05 \x03(\x0b\x32\x30.cirq.google.api.v2.InternalGate.CustomArgsEntry\x1aH\n\rGateArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x17.cirq.google.api.v2.Arg:\x02\x38\x01\x1aP\n\x0f\x43ustomArgsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12,\n\x05value\x18\x02 \x01(\x0b\x32\x1d.cirq.google.api.v2.CustomArg:\x02\x38\x01\"\xd8\x03\n\x10\x43ouplerPulseGate\x12\x37\n\x0chold_time_ps\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x00\x88\x01\x01\x12\x37\n\x0crise_time_ps\x18\x02 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x01\x88\x01\x01\x12:\n\x0fpadding_time_ps\x18\x03 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x02\x88\x01\x01\x12\x37\n\x0c\x63oupling_mhz\x18\x04 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x03\x88\x01\x01\x12\x38\n\rq0_detune_mhz\x18\x05 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x04\x88\x01\x01\x12\x38\n\rq1_detune_mhz\x18\x06 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgH\x05\x88\x01\x01\x42\x0f\n\r_hold_time_psB\x0f\n\r_rise_time_psB\x12\n\x10_padding_time_psB\x0f\n\r_coupling_mhzB\x10\n\x0e_q0_detune_mhzB\x10\n\x0e_q1_detune_mhz\"\x8b\x01\n\x0f\x43liffordTableau\x12\x17\n\nnum_qubits\x18\x01 \x01(\x05H\x00\x88\x01\x01\x12\x1a\n\rinitial_state\x18\x02 \x01(\x05H\x01\x88\x01\x01\x12\n\n\x02rs\x18\x03 \x03(\x08\x12\n\n\x02xs\x18\x04 \x03(\x08\x12\n\n\x02zs\x18\x05 \x03(\x08\x42\r\n\x0b_num_qubitsB\x10\n\x0e_initial_state\"O\n\x17SingleQubitCliffordGate\x12\x34\n\x07tableau\x18\x01 \x01(\x0b\x32#.cirq.google.api.v2.CliffordTableau\"!\n\x0cIdentityGate\x12\x11\n\tqid_shape\x18\x01 \x03(\r\":\n\x08HPowGate\x12.\n\x08\x65xponent\x18\x01 \x01(\x0b\x32\x1c.cirq.google.api.v2.FloatArgB/\n\x1d\x63om.google.cirq.google.api.v2B\x0cProgramProtoP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'cirq_google.api.v2.program_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\035com.google.cirq.google.api.v2B\014ProgramProtoP\001'
  _globals['_LANGUAGE'].fields_by_name['gate_set']._options = None
  _globals['_LANGUAGE'].fields_by_name['gate_set']._serialized_options = b'\030\001'
  _globals['_OPERATION_ARGSENTRY']._options = None
  _globals['_OPERATION_ARGSENTRY']._serialized_options = b'8\001'
  _globals['_OPERATION'].fields_by_name['gate']._options = None
  _globals['_OPERATION'].fields_by_name['gate']._serialized_options = b'\030\001'
  _globals['_OPERATION'].fields_by_name['args']._options = None
  _globals['_OPERATION'].fields_by_name['args']._serialized_options = b'\030\001'
  _globals['_FUNCTIONINTERPOLATION'].fields_by_name['x_values']._options = None
  _globals['_FUNCTIONINTERPOLATION'].fields_by_name['x_values']._serialized_options = b'\020\001'
  _globals['_FUNCTIONINTERPOLATION'].fields_by_name['y_values']._options = None
  _globals['_FUNCTIONINTERPOLATION'].fields_by_name['y_values']._serialized_options = b'\020\001'
  _globals['_INTERNALGATE_GATEARGSENTRY']._options = None
  _globals['_INTERNALGATE_GATEARGSENTRY']._serialized_options = b'8\001'
  _globals['_INTERNALGATE_CUSTOMARGSENTRY']._options = None
  _globals['_INTERNALGATE_CUSTOMARGSENTRY']._serialized_options = b'8\001'
  _globals['_PROGRAM']._serialized_start=84
  _globals['_PROGRAM']._serialized_end=299
  _globals['_CONSTANT']._serialized_start=302
  _globals['_CONSTANT']._serialized_end=501
  _globals['_CIRCUIT']._serialized_start=504
  _globals['_CIRCUIT']._serialized_end=716
  _globals['_CIRCUIT_SCHEDULINGSTRATEGY']._serialized_start=637
  _globals['_CIRCUIT_SCHEDULINGSTRATEGY']._serialized_end=716
  _globals['_MOMENT']._serialized_start=719
  _globals['_MOMENT']._serialized_end=906
  _globals['_SCHEDULE']._serialized_start=908
  _globals['_SCHEDULE']._serialized_end=988
  _globals['_SCHEDULEDOPERATION']._serialized_start=990
  _globals['_SCHEDULEDOPERATION']._serialized_end=1086
  _globals['_LANGUAGE']._serialized_start=1088
  _globals['_LANGUAGE']._serialized_end=1151
  _globals['_FLOATARG']._serialized_start=1153
  _globals['_FLOATARG']._serialized_end=1260
  _globals['_XPOWGATE']._serialized_start=1262
  _globals['_XPOWGATE']._serialized_end=1320
  _globals['_YPOWGATE']._serialized_start=1322
  _globals['_YPOWGATE']._serialized_end=1380
  _globals['_ZPOWGATE']._serialized_start=1382
  _globals['_ZPOWGATE']._serialized_end=1463
  _globals['_PHASEDXPOWGATE']._serialized_start=1465
  _globals['_PHASEDXPOWGATE']._serialized_end=1583
  _globals['_PHASEDXZGATE']._serialized_start=1586
  _globals['_PHASEDXZGATE']._serialized_end=1759
  _globals['_CZPOWGATE']._serialized_start=1761
  _globals['_CZPOWGATE']._serialized_end=1820
  _globals['_FSIMGATE']._serialized_start=1822
  _globals['_FSIMGATE']._serialized_end=1949
  _globals['_ISWAPPOWGATE']._serialized_start=1951
  _globals['_ISWAPPOWGATE']._serialized_end=2013
  _globals['_MEASUREMENTGATE']._serialized_start=2015
  _globals['_MEASUREMENTGATE']._serialized_end=2116
  _globals['_WAITGATE']._serialized_start=2118
  _globals['_WAITGATE']._serialized_end=2182
  _globals['_OPERATION']._serialized_start=2185
  _globals['_OPERATION']._serialized_end=3415
  _globals['_OPERATION_ARGSENTRY']._serialized_start=3324
  _globals['_OPERATION_ARGSENTRY']._serialized_end=3392
  _globals['_DYNAMICALDECOUPLINGTAG']._serialized_start=3417
  _globals['_DYNAMICALDECOUPLINGTAG']._serialized_end=3477
  _globals['_TAG']._serialized_start=3479
  _globals['_TAG']._serialized_end=3567
  _globals['_GATE']._serialized_start=3569
  _globals['_GATE']._serialized_end=3587
  _globals['_QUBIT']._serialized_start=3589
  _globals['_QUBIT']._serialized_end=3608
  _globals['_ARG']._serialized_start=3611
  _globals['_ARG']._serialized_end=3767
  _globals['_ARGVALUE']._serialized_start=3770
  _globals['_ARGVALUE']._serialized_end=4147
  _globals['_REPEATEDINT64']._serialized_start=4149
  _globals['_REPEATEDINT64']._serialized_end=4180
  _globals['_REPEATEDDOUBLE']._serialized_start=4182
  _globals['_REPEATEDDOUBLE']._serialized_end=4214
  _globals['_REPEATEDSTRING']._serialized_start=4216
  _globals['_REPEATEDSTRING']._serialized_end=4248
  _globals['_REPEATEDBOOLEAN']._serialized_start=4250
  _globals['_REPEATEDBOOLEAN']._serialized_end=4283
  _globals['_ARGFUNCTION']._serialized_start=4285
  _globals['_ARGFUNCTION']._serialized_end=4351
  _globals['_CIRCUITOPERATION']._serialized_start=4354
  _globals['_CIRCUITOPERATION']._serialized_end=4657
  _globals['_REPETITIONSPECIFICATION']._serialized_start=4660
  _globals['_REPETITIONSPECIFICATION']._serialized_end=4848
  _globals['_REPETITIONSPECIFICATION_REPETITIONIDS']._serialized_start=4800
  _globals['_REPETITIONSPECIFICATION_REPETITIONIDS']._serialized_end=4828
  _globals['_QUBITMAPPING']._serialized_start=4851
  _globals['_QUBITMAPPING']._serialized_end=5023
  _globals['_QUBITMAPPING_QUBITENTRY']._serialized_start=4929
  _globals['_QUBITMAPPING_QUBITENTRY']._serialized_end=5023
  _globals['_MEASUREMENTKEY']._serialized_start=5025
  _globals['_MEASUREMENTKEY']._serialized_end=5061
  _globals['_MEASUREMENTKEYMAPPING']._serialized_start=5064
  _globals['_MEASUREMENTKEYMAPPING']._serialized_end=5290
  _globals['_MEASUREMENTKEYMAPPING_MEASUREMENTKEYENTRY']._serialized_start=5169
  _globals['_MEASUREMENTKEYMAPPING_MEASUREMENTKEYENTRY']._serialized_end=5290
  _globals['_ARGMAPPING']._serialized_start=5293
  _globals['_ARGMAPPING']._serialized_end=5453
  _globals['_ARGMAPPING_ARGENTRY']._serialized_start=5365
  _globals['_ARGMAPPING_ARGENTRY']._serialized_end=5453
  _globals['_FUNCTIONINTERPOLATION']._serialized_start=5455
  _globals['_FUNCTIONINTERPOLATION']._serialized_end=5522
  _globals['_CUSTOMARG']._serialized_start=5524
  _globals['_CUSTOMARG']._serialized_end=5631
  _globals['_INTERNALGATE']._serialized_start=5634
  _globals['_INTERNALGATE']._serialized_end=5992
  _globals['_INTERNALGATE_GATEARGSENTRY']._serialized_start=5838
  _globals['_INTERNALGATE_GATEARGSENTRY']._serialized_end=5910
  _globals['_INTERNALGATE_CUSTOMARGSENTRY']._serialized_start=5912
  _globals['_INTERNALGATE_CUSTOMARGSENTRY']._serialized_end=5992
  _globals['_COUPLERPULSEGATE']._serialized_start=5995
  _globals['_COUPLERPULSEGATE']._serialized_end=6467
  _globals['_CLIFFORDTABLEAU']._serialized_start=6470
  _globals['_CLIFFORDTABLEAU']._serialized_end=6609
  _globals['_SINGLEQUBITCLIFFORDGATE']._serialized_start=6611
  _globals['_SINGLEQUBITCLIFFORDGATE']._serialized_end=6690
  _globals['_IDENTITYGATE']._serialized_start=6692
  _globals['_IDENTITYGATE']._serialized_end=6725
  _globals['_HPOWGATE']._serialized_start=6727
  _globals['_HPOWGATE']._serialized_end=6785
# @@protoc_insertion_point(module_scope)
