# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: cirq_google/api/v1/params.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'cirq_google/api/v1/params.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1f\x63irq_google/api/v1/params.proto\x12\x12\x63irq.google.api.v1\"V\n\x0eParameterSweep\x12\x13\n\x0brepetitions\x18\x01 \x01(\x05\x12/\n\x05sweep\x18\x02 \x01(\x0b\x32 .cirq.google.api.v1.ProductSweep\"=\n\x0cProductSweep\x12-\n\x07\x66\x61\x63tors\x18\x01 \x03(\x0b\x32\x1c.cirq.google.api.v1.ZipSweep\";\n\x08ZipSweep\x12/\n\x06sweeps\x18\x01 \x03(\x0b\x32\x1f.cirq.google.api.v1.SingleSweep\"\x8d\x01\n\x0bSingleSweep\x12\x15\n\rparameter_key\x18\x01 \x01(\t\x12,\n\x06points\x18\x02 \x01(\x0b\x32\x1a.cirq.google.api.v1.PointsH\x00\x12\x30\n\x08linspace\x18\x03 \x01(\x0b\x32\x1c.cirq.google.api.v1.LinspaceH\x00\x42\x07\n\x05sweep\"\x18\n\x06Points\x12\x0e\n\x06points\x18\x01 \x03(\x02\"G\n\x08Linspace\x12\x13\n\x0b\x66irst_point\x18\x01 \x01(\x02\x12\x12\n\nlast_point\x18\x02 \x01(\x02\x12\x12\n\nnum_points\x18\x03 \x01(\x03\"\x8c\x01\n\rParameterDict\x12G\n\x0b\x61ssignments\x18\x01 \x03(\x0b\x32\x32.cirq.google.api.v1.ParameterDict.AssignmentsEntry\x1a\x32\n\x10\x41ssignmentsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\x42.\n\x1d\x63om.google.cirq.google.api.v1B\x0bParamsProtoP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'cirq_google.api.v1.params_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\035com.google.cirq.google.api.v1B\013ParamsProtoP\001'
  _globals['_PARAMETERDICT_ASSIGNMENTSENTRY']._loaded_options = None
  _globals['_PARAMETERDICT_ASSIGNMENTSENTRY']._serialized_options = b'8\001'
  _globals['_PARAMETERSWEEP']._serialized_start=55
  _globals['_PARAMETERSWEEP']._serialized_end=141
  _globals['_PRODUCTSWEEP']._serialized_start=143
  _globals['_PRODUCTSWEEP']._serialized_end=204
  _globals['_ZIPSWEEP']._serialized_start=206
  _globals['_ZIPSWEEP']._serialized_end=265
  _globals['_SINGLESWEEP']._serialized_start=268
  _globals['_SINGLESWEEP']._serialized_end=409
  _globals['_POINTS']._serialized_start=411
  _globals['_POINTS']._serialized_end=435
  _globals['_LINSPACE']._serialized_start=437
  _globals['_LINSPACE']._serialized_end=508
  _globals['_PARAMETERDICT']._serialized_start=511
  _globals['_PARAMETERDICT']._serialized_end=651
  _globals['_PARAMETERDICT_ASSIGNMENTSENTRY']._serialized_start=601
  _globals['_PARAMETERDICT_ASSIGNMENTSENTRY']._serialized_end=651
# @@protoc_insertion_point(module_scope)
