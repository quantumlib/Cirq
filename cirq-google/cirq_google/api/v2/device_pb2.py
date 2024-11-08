# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: cirq_google/api/v2/device.proto
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
    'cirq_google/api/v2/device.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1f\x63irq_google/api/v2/device.proto\x12\x12\x63irq.google.api.v2\"\xfa\x01\n\x13\x44\x65viceSpecification\x12\x38\n\x0fvalid_gate_sets\x18\x01 \x03(\x0b\x32\x1b.cirq.google.api.v2.GateSetB\x02\x18\x01\x12:\n\x0bvalid_gates\x18\x05 \x03(\x0b\x32%.cirq.google.api.v2.GateSpecification\x12\x14\n\x0cvalid_qubits\x18\x02 \x03(\t\x12\x34\n\rvalid_targets\x18\x03 \x03(\x0b\x32\x1d.cirq.google.api.v2.TargetSet\x12!\n\x19\x64\x65veloper_recommendations\x18\x04 \x01(\t\"\xfe\x08\n\x11GateSpecification\x12\x1b\n\x13gate_duration_picos\x18\x01 \x01(\x03\x12=\n\x03syc\x18\x02 \x01(\x0b\x32..cirq.google.api.v2.GateSpecification.SycamoreH\x00\x12\x45\n\nsqrt_iswap\x18\x03 \x01(\x0b\x32/.cirq.google.api.v2.GateSpecification.SqrtISwapH\x00\x12L\n\x0esqrt_iswap_inv\x18\x04 \x01(\x0b\x32\x32.cirq.google.api.v2.GateSpecification.SqrtISwapInvH\x00\x12\x36\n\x02\x63z\x18\x05 \x01(\x0b\x32(.cirq.google.api.v2.GateSpecification.CZH\x00\x12\x43\n\tphased_xz\x18\x06 \x01(\x0b\x32..cirq.google.api.v2.GateSpecification.PhasedXZH\x00\x12I\n\x0cvirtual_zpow\x18\x07 \x01(\x0b\x32\x31.cirq.google.api.v2.GateSpecification.VirtualZPowH\x00\x12K\n\rphysical_zpow\x18\x08 \x01(\x0b\x32\x32.cirq.google.api.v2.GateSpecification.PhysicalZPowH\x00\x12K\n\rcoupler_pulse\x18\t \x01(\x0b\x32\x32.cirq.google.api.v2.GateSpecification.CouplerPulseH\x00\x12\x41\n\x04meas\x18\n \x01(\x0b\x32\x31.cirq.google.api.v2.GateSpecification.MeasurementH\x00\x12:\n\x04wait\x18\x0b \x01(\x0b\x32*.cirq.google.api.v2.GateSpecification.WaitH\x00\x12L\n\x0e\x66sim_via_model\x18\x0c \x01(\x0b\x32\x32.cirq.google.api.v2.GateSpecification.FSimViaModelH\x00\x12\x46\n\x0b\x63z_pow_gate\x18\r \x01(\x0b\x32/.cirq.google.api.v2.GateSpecification.CZPowGateH\x00\x12K\n\rinternal_gate\x18\x0e \x01(\x0b\x32\x32.cirq.google.api.v2.GateSpecification.InternalGateH\x00\x1a\n\n\x08Sycamore\x1a\x0b\n\tSqrtISwap\x1a\x0e\n\x0cSqrtISwapInv\x1a\x04\n\x02\x43Z\x1a\n\n\x08PhasedXZ\x1a\r\n\x0bVirtualZPow\x1a\x0e\n\x0cPhysicalZPow\x1a\x0e\n\x0c\x43ouplerPulse\x1a\r\n\x0bMeasurement\x1a\x06\n\x04Wait\x1a\x0e\n\x0c\x46SimViaModel\x1a\x0b\n\tCZPowGate\x1a\x0e\n\x0cInternalGateB\x06\n\x04gate\"P\n\x07GateSet\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x37\n\x0bvalid_gates\x18\x02 \x03(\x0b\x32\".cirq.google.api.v2.GateDefinition\"\xa1\x01\n\x0eGateDefinition\x12\n\n\x02id\x18\x01 \x01(\t\x12\x18\n\x10number_of_qubits\x18\x02 \x01(\x05\x12\x35\n\nvalid_args\x18\x03 \x03(\x0b\x32!.cirq.google.api.v2.ArgDefinition\x12\x1b\n\x13gate_duration_picos\x18\x04 \x01(\x03\x12\x15\n\rvalid_targets\x18\x05 \x03(\t\"\xda\x01\n\rArgDefinition\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x37\n\x04type\x18\x02 \x01(\x0e\x32).cirq.google.api.v2.ArgDefinition.ArgType\x12\x39\n\x0e\x61llowed_ranges\x18\x03 \x03(\x0b\x32!.cirq.google.api.v2.ArgumentRange\"G\n\x07\x41rgType\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\t\n\x05\x46LOAT\x10\x01\x12\x14\n\x10REPEATED_BOOLEAN\x10\x02\x12\n\n\x06STRING\x10\x03\"=\n\rArgumentRange\x12\x15\n\rminimum_value\x18\x01 \x01(\x02\x12\x15\n\rmaximum_value\x18\x02 \x01(\x02\"\xef\x01\n\tTargetSet\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x45\n\x0ftarget_ordering\x18\x02 \x01(\x0e\x32,.cirq.google.api.v2.TargetSet.TargetOrdering\x12+\n\x07targets\x18\x03 \x03(\x0b\x32\x1a.cirq.google.api.v2.Target\"`\n\x0eTargetOrdering\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\r\n\tSYMMETRIC\x10\x01\x12\x12\n\nASYMMETRIC\x10\x02\x1a\x02\x08\x01\x12\x1a\n\x12SUBSET_PERMUTATION\x10\x03\x1a\x02\x08\x01\"\x15\n\x06Target\x12\x0b\n\x03ids\x18\x01 \x03(\tB.\n\x1d\x63om.google.cirq.google.api.v2B\x0b\x44\x65viceProtoP\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'cirq_google.api.v2.device_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\035com.google.cirq.google.api.v2B\013DeviceProtoP\001'
  _globals['_DEVICESPECIFICATION'].fields_by_name['valid_gate_sets']._loaded_options = None
  _globals['_DEVICESPECIFICATION'].fields_by_name['valid_gate_sets']._serialized_options = b'\030\001'
  _globals['_TARGETSET_TARGETORDERING'].values_by_name["ASYMMETRIC"]._loaded_options = None
  _globals['_TARGETSET_TARGETORDERING'].values_by_name["ASYMMETRIC"]._serialized_options = b'\010\001'
  _globals['_TARGETSET_TARGETORDERING'].values_by_name["SUBSET_PERMUTATION"]._loaded_options = None
  _globals['_TARGETSET_TARGETORDERING'].values_by_name["SUBSET_PERMUTATION"]._serialized_options = b'\010\001'
  _globals['_DEVICESPECIFICATION']._serialized_start=56
  _globals['_DEVICESPECIFICATION']._serialized_end=306
  _globals['_GATESPECIFICATION']._serialized_start=309
  _globals['_GATESPECIFICATION']._serialized_end=1459
  _globals['_GATESPECIFICATION_SYCAMORE']._serialized_start=1279
  _globals['_GATESPECIFICATION_SYCAMORE']._serialized_end=1289
  _globals['_GATESPECIFICATION_SQRTISWAP']._serialized_start=1291
  _globals['_GATESPECIFICATION_SQRTISWAP']._serialized_end=1302
  _globals['_GATESPECIFICATION_SQRTISWAPINV']._serialized_start=1304
  _globals['_GATESPECIFICATION_SQRTISWAPINV']._serialized_end=1318
  _globals['_GATESPECIFICATION_CZ']._serialized_start=1320
  _globals['_GATESPECIFICATION_CZ']._serialized_end=1324
  _globals['_GATESPECIFICATION_PHASEDXZ']._serialized_start=1326
  _globals['_GATESPECIFICATION_PHASEDXZ']._serialized_end=1336
  _globals['_GATESPECIFICATION_VIRTUALZPOW']._serialized_start=1338
  _globals['_GATESPECIFICATION_VIRTUALZPOW']._serialized_end=1351
  _globals['_GATESPECIFICATION_PHYSICALZPOW']._serialized_start=1353
  _globals['_GATESPECIFICATION_PHYSICALZPOW']._serialized_end=1367
  _globals['_GATESPECIFICATION_COUPLERPULSE']._serialized_start=1369
  _globals['_GATESPECIFICATION_COUPLERPULSE']._serialized_end=1383
  _globals['_GATESPECIFICATION_MEASUREMENT']._serialized_start=1385
  _globals['_GATESPECIFICATION_MEASUREMENT']._serialized_end=1398
  _globals['_GATESPECIFICATION_WAIT']._serialized_start=1400
  _globals['_GATESPECIFICATION_WAIT']._serialized_end=1406
  _globals['_GATESPECIFICATION_FSIMVIAMODEL']._serialized_start=1408
  _globals['_GATESPECIFICATION_FSIMVIAMODEL']._serialized_end=1422
  _globals['_GATESPECIFICATION_CZPOWGATE']._serialized_start=1424
  _globals['_GATESPECIFICATION_CZPOWGATE']._serialized_end=1435
  _globals['_GATESPECIFICATION_INTERNALGATE']._serialized_start=1437
  _globals['_GATESPECIFICATION_INTERNALGATE']._serialized_end=1451
  _globals['_GATESET']._serialized_start=1461
  _globals['_GATESET']._serialized_end=1541
  _globals['_GATEDEFINITION']._serialized_start=1544
  _globals['_GATEDEFINITION']._serialized_end=1705
  _globals['_ARGDEFINITION']._serialized_start=1708
  _globals['_ARGDEFINITION']._serialized_end=1926
  _globals['_ARGDEFINITION_ARGTYPE']._serialized_start=1855
  _globals['_ARGDEFINITION_ARGTYPE']._serialized_end=1926
  _globals['_ARGUMENTRANGE']._serialized_start=1928
  _globals['_ARGUMENTRANGE']._serialized_end=1989
  _globals['_TARGETSET']._serialized_start=1992
  _globals['_TARGETSET']._serialized_end=2231
  _globals['_TARGETSET_TARGETORDERING']._serialized_start=2135
  _globals['_TARGETSET_TARGETORDERING']._serialized_end=2231
  _globals['_TARGET']._serialized_start=2233
  _globals['_TARGET']._serialized_end=2254
# @@protoc_insertion_point(module_scope)
