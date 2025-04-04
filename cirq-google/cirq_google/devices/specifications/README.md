# Device specifications of Google quantum processors

This directory contains snapshots of `DeviceSpecification` proto messages
(defined in `cirq-google/cirq_google/api/v2/device.proto`) describing Google
devices.

Files with the suffix `_for_grid_device` are equivalent representations of
corresponding proto files without the suffix, but in the new
`DeviceSpecification` format which is parsed into `cirq_google.GridDevice`.
