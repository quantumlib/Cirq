# Copyright 2024 The Cirq Developers
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

from cirq_google.api.v2 import program_pb2
from cirq_google.api.v2 import run_context_pb2
import cirq_google.api.v2.run_context as run_context
import google.protobuf.text_format as text_format


def test_converting_multiple_device_params_to_device_parameters_diff() -> None:
    """Test of converting a list of DeviceParameter's to a DeviceParametersDiff object."""
    readout_paths = (["q3_4", "readout_default"], ["q5_6", "readout_default"])

    device_params = []
    for readout_path in readout_paths:
        device_params.extend(
            [
                (
                    run_context_pb2.DeviceParameter(
                        path=[*readout_path, "readoutDemodDelay"], units="ns"
                    ),
                    program_pb2.ArgValue(float_value=5.0),
                ),
                (
                    run_context_pb2.DeviceParameter(path=[*readout_path, "readoutFidelities"]),
                    program_pb2.ArgValue(
                        double_values=program_pb2.RepeatedDouble(values=[0.991, 0.993])
                    ),
                ),
                (
                    run_context_pb2.DeviceParameter(path=[*readout_path, "demod", "phase_i_rad"]),
                    program_pb2.ArgValue(double_value=0.0),
                ),
            ]
        )
    diff = run_context.to_device_parameters_diff(device_params)
    expected_diff_pb_text = """
        groups {
          parent: -1
        }
        groups {
          parent: 0
          name: 1
        }
        groups {
          parent: 1
          name: 4
        }
        groups {
          parent: -1
          name: 6
        }
        groups {
          parent: 3
          name: 1
        }
        groups {
          parent: 4
          name: 4
        }
        params {
          resource_group: 1
          name: 2
          value {
            float_value: 5
          }
        }
        params {
          resource_group: 1
          name: 3
          value {
            double_values {
              values: 0.991
              values: 0.993
            }
          }
        }
        params {
          resource_group: 2
          name: 5
          value {
            double_value: 0
          }
        }
        params {
          resource_group: 4
          name: 2
          value {
            float_value: 5
          }
        }
        params {
          resource_group: 4
          name: 3
          value {
            double_values {
              values: 0.991
              values: 0.993
            }
          }
        }
        params {
          resource_group: 5
          name: 5
          value {
            double_value: 0
          }
        }
        strs: "q3_4"
        strs: "readout_default"
        strs: "readoutDemodDelay"
        strs: "readoutFidelities"
        strs: "demod"
        strs: "phase_i_rad"
        strs: "q5_6"
    """
    print(diff)
    assert text_format.Parse(expected_diff_pb_text, run_context_pb2.DeviceParametersDiff()) == diff


def test_converting_to_device_parameters_diff_token_id_caching_is_correct() -> None:
    """Test that multiple calling of run_context.to_device_parameters_diff gives
    correct token id assignments.
    """
    device_params = [
        (
            run_context_pb2.DeviceParameter(
                path=["q1_2", "readout_default", "readoutDemodDelay"], units="ns"
            ),
            program_pb2.ArgValue(float_value=5.0),
        )
    ]

    diff = run_context.to_device_parameters_diff(device_params)
    expected_diff_pb_text = """
        groups {
          parent: -1
        }
        groups {
          name: 1
        }
        params {
          resource_group: 1
          name: 2
          value {
            float_value: 5
          }
        }
        strs: "q1_2"
        strs: "readout_default"
        strs: "readoutDemodDelay"
    """
    assert text_format.Parse(expected_diff_pb_text, run_context_pb2.DeviceParametersDiff()) == diff
