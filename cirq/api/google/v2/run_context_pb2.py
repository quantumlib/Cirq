# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: cirq/api/google/v2/run_context.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='cirq/api/google/v2/run_context.proto',
  package='cirq.api.google.v2',
  syntax='proto3',
  serialized_options=_b('\n\035com.google.cirq.api.google.v2B\017RunContextProtoP\001'),
  serialized_pb=_b('\n$cirq/api/google/v2/run_context.proto\x12\x12\x63irq.api.google.v2\"J\n\nRunContext\x12<\n\x10parameter_sweeps\x18\x01 \x03(\x0b\x32\".cirq.api.google.v2.ParameterSweep\"O\n\x0eParameterSweep\x12\x13\n\x0brepetitions\x18\x01 \x01(\x05\x12(\n\x05sweep\x18\x02 \x01(\x0b\x32\x19.cirq.api.google.v2.Sweep\"\x86\x01\n\x05Sweep\x12;\n\x0esweep_function\x18\x01 \x01(\x0b\x32!.cirq.api.google.v2.SweepFunctionH\x00\x12\x37\n\x0csingle_sweep\x18\x02 \x01(\x0b\x32\x1f.cirq.api.google.v2.SingleSweepH\x00\x42\x07\n\x05sweep\"\xc6\x01\n\rSweepFunction\x12\x45\n\rfunction_type\x18\x01 \x01(\x0e\x32..cirq.api.google.v2.SweepFunction.FunctionType\x12)\n\x06sweeps\x18\x02 \x03(\x0b\x32\x19.cirq.api.google.v2.Sweep\"C\n\x0c\x46unctionType\x12\x1d\n\x19\x46UNCTION_TYPE_UNSPECIFIED\x10\x00\x12\x0b\n\x07PRODUCT\x10\x01\x12\x07\n\x03ZIP\x10\x02\"\x8d\x01\n\x0bSingleSweep\x12\x15\n\rparameter_key\x18\x01 \x01(\t\x12,\n\x06points\x18\x02 \x01(\x0b\x32\x1a.cirq.api.google.v2.PointsH\x00\x12\x30\n\x08linspace\x18\x03 \x01(\x0b\x32\x1c.cirq.api.google.v2.LinspaceH\x00\x42\x07\n\x05sweep\"\x18\n\x06Points\x12\x0e\n\x06points\x18\x01 \x03(\x02\"G\n\x08Linspace\x12\x13\n\x0b\x66irst_point\x18\x01 \x01(\x02\x12\x12\n\nlast_point\x18\x02 \x01(\x02\x12\x12\n\nnum_points\x18\x03 \x01(\x03\x42\x32\n\x1d\x63om.google.cirq.api.google.v2B\x0fRunContextProtoP\x01\x62\x06proto3')
)



_SWEEPFUNCTION_FUNCTIONTYPE = _descriptor.EnumDescriptor(
  name='FunctionType',
  full_name='cirq.api.google.v2.SweepFunction.FunctionType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FUNCTION_TYPE_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRODUCT', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ZIP', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=486,
  serialized_end=553,
)
_sym_db.RegisterEnumDescriptor(_SWEEPFUNCTION_FUNCTIONTYPE)


_RUNCONTEXT = _descriptor.Descriptor(
  name='RunContext',
  full_name='cirq.api.google.v2.RunContext',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='parameter_sweeps', full_name='cirq.api.google.v2.RunContext.parameter_sweeps', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=134,
)


_PARAMETERSWEEP = _descriptor.Descriptor(
  name='ParameterSweep',
  full_name='cirq.api.google.v2.ParameterSweep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='repetitions', full_name='cirq.api.google.v2.ParameterSweep.repetitions', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sweep', full_name='cirq.api.google.v2.ParameterSweep.sweep', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=215,
)


_SWEEP = _descriptor.Descriptor(
  name='Sweep',
  full_name='cirq.api.google.v2.Sweep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sweep_function', full_name='cirq.api.google.v2.Sweep.sweep_function', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='single_sweep', full_name='cirq.api.google.v2.Sweep.single_sweep', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='sweep', full_name='cirq.api.google.v2.Sweep.sweep',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=218,
  serialized_end=352,
)


_SWEEPFUNCTION = _descriptor.Descriptor(
  name='SweepFunction',
  full_name='cirq.api.google.v2.SweepFunction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='function_type', full_name='cirq.api.google.v2.SweepFunction.function_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sweeps', full_name='cirq.api.google.v2.SweepFunction.sweeps', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SWEEPFUNCTION_FUNCTIONTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=355,
  serialized_end=553,
)


_SINGLESWEEP = _descriptor.Descriptor(
  name='SingleSweep',
  full_name='cirq.api.google.v2.SingleSweep',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='parameter_key', full_name='cirq.api.google.v2.SingleSweep.parameter_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='points', full_name='cirq.api.google.v2.SingleSweep.points', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='linspace', full_name='cirq.api.google.v2.SingleSweep.linspace', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='sweep', full_name='cirq.api.google.v2.SingleSweep.sweep',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=556,
  serialized_end=697,
)


_POINTS = _descriptor.Descriptor(
  name='Points',
  full_name='cirq.api.google.v2.Points',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='points', full_name='cirq.api.google.v2.Points.points', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=699,
  serialized_end=723,
)


_LINSPACE = _descriptor.Descriptor(
  name='Linspace',
  full_name='cirq.api.google.v2.Linspace',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='first_point', full_name='cirq.api.google.v2.Linspace.first_point', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='last_point', full_name='cirq.api.google.v2.Linspace.last_point', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_points', full_name='cirq.api.google.v2.Linspace.num_points', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=725,
  serialized_end=796,
)

_RUNCONTEXT.fields_by_name['parameter_sweeps'].message_type = _PARAMETERSWEEP
_PARAMETERSWEEP.fields_by_name['sweep'].message_type = _SWEEP
_SWEEP.fields_by_name['sweep_function'].message_type = _SWEEPFUNCTION
_SWEEP.fields_by_name['single_sweep'].message_type = _SINGLESWEEP
_SWEEP.oneofs_by_name['sweep'].fields.append(
  _SWEEP.fields_by_name['sweep_function'])
_SWEEP.fields_by_name['sweep_function'].containing_oneof = _SWEEP.oneofs_by_name['sweep']
_SWEEP.oneofs_by_name['sweep'].fields.append(
  _SWEEP.fields_by_name['single_sweep'])
_SWEEP.fields_by_name['single_sweep'].containing_oneof = _SWEEP.oneofs_by_name['sweep']
_SWEEPFUNCTION.fields_by_name['function_type'].enum_type = _SWEEPFUNCTION_FUNCTIONTYPE
_SWEEPFUNCTION.fields_by_name['sweeps'].message_type = _SWEEP
_SWEEPFUNCTION_FUNCTIONTYPE.containing_type = _SWEEPFUNCTION
_SINGLESWEEP.fields_by_name['points'].message_type = _POINTS
_SINGLESWEEP.fields_by_name['linspace'].message_type = _LINSPACE
_SINGLESWEEP.oneofs_by_name['sweep'].fields.append(
  _SINGLESWEEP.fields_by_name['points'])
_SINGLESWEEP.fields_by_name['points'].containing_oneof = _SINGLESWEEP.oneofs_by_name['sweep']
_SINGLESWEEP.oneofs_by_name['sweep'].fields.append(
  _SINGLESWEEP.fields_by_name['linspace'])
_SINGLESWEEP.fields_by_name['linspace'].containing_oneof = _SINGLESWEEP.oneofs_by_name['sweep']
DESCRIPTOR.message_types_by_name['RunContext'] = _RUNCONTEXT
DESCRIPTOR.message_types_by_name['ParameterSweep'] = _PARAMETERSWEEP
DESCRIPTOR.message_types_by_name['Sweep'] = _SWEEP
DESCRIPTOR.message_types_by_name['SweepFunction'] = _SWEEPFUNCTION
DESCRIPTOR.message_types_by_name['SingleSweep'] = _SINGLESWEEP
DESCRIPTOR.message_types_by_name['Points'] = _POINTS
DESCRIPTOR.message_types_by_name['Linspace'] = _LINSPACE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RunContext = _reflection.GeneratedProtocolMessageType('RunContext', (_message.Message,), dict(
  DESCRIPTOR = _RUNCONTEXT,
  __module__ = 'cirq.api.google.v2.run_context_pb2'
  # @@protoc_insertion_point(class_scope:cirq.api.google.v2.RunContext)
  ))
_sym_db.RegisterMessage(RunContext)

ParameterSweep = _reflection.GeneratedProtocolMessageType('ParameterSweep', (_message.Message,), dict(
  DESCRIPTOR = _PARAMETERSWEEP,
  __module__ = 'cirq.api.google.v2.run_context_pb2'
  # @@protoc_insertion_point(class_scope:cirq.api.google.v2.ParameterSweep)
  ))
_sym_db.RegisterMessage(ParameterSweep)

Sweep = _reflection.GeneratedProtocolMessageType('Sweep', (_message.Message,), dict(
  DESCRIPTOR = _SWEEP,
  __module__ = 'cirq.api.google.v2.run_context_pb2'
  # @@protoc_insertion_point(class_scope:cirq.api.google.v2.Sweep)
  ))
_sym_db.RegisterMessage(Sweep)

SweepFunction = _reflection.GeneratedProtocolMessageType('SweepFunction', (_message.Message,), dict(
  DESCRIPTOR = _SWEEPFUNCTION,
  __module__ = 'cirq.api.google.v2.run_context_pb2'
  # @@protoc_insertion_point(class_scope:cirq.api.google.v2.SweepFunction)
  ))
_sym_db.RegisterMessage(SweepFunction)

SingleSweep = _reflection.GeneratedProtocolMessageType('SingleSweep', (_message.Message,), dict(
  DESCRIPTOR = _SINGLESWEEP,
  __module__ = 'cirq.api.google.v2.run_context_pb2'
  # @@protoc_insertion_point(class_scope:cirq.api.google.v2.SingleSweep)
  ))
_sym_db.RegisterMessage(SingleSweep)

Points = _reflection.GeneratedProtocolMessageType('Points', (_message.Message,), dict(
  DESCRIPTOR = _POINTS,
  __module__ = 'cirq.api.google.v2.run_context_pb2'
  # @@protoc_insertion_point(class_scope:cirq.api.google.v2.Points)
  ))
_sym_db.RegisterMessage(Points)

Linspace = _reflection.GeneratedProtocolMessageType('Linspace', (_message.Message,), dict(
  DESCRIPTOR = _LINSPACE,
  __module__ = 'cirq.api.google.v2.run_context_pb2'
  # @@protoc_insertion_point(class_scope:cirq.api.google.v2.Linspace)
  ))
_sym_db.RegisterMessage(Linspace)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
