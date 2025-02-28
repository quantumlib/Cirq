# Copyright 2023 The Cirq Developers
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
import cirq_google


def test_parameters():
    param = cirq_google.study.DeviceParameter(
        path=('test', 'subdir'), idx=2, value='tmp', units='GHz'
    )
    cirq.testing.assert_equivalent_repr(param, global_vals={'cirq_google': cirq_google})


def test_metadata():
    param = cirq_google.study.DeviceParameter(path=('test', 'subdir'), idx=2, value='tmp')
    metadata = cirq_google.study.Metadata(
        device_parameters=[param], is_const=True, label="fake_label"
    )
    cirq.testing.assert_equivalent_repr(metadata, global_vals={'cirq_google': cirq_google})


def test_metadata_json_roundtrip():
    metadata = cirq_google.study.Metadata(
        device_parameters=[cirq_google.study.DeviceParameter(path=['test', 'subdir'])],
        is_const=True,
        label="fake_label",
    )
    metadata_reconstruct = cirq.read_json(json_text=cirq.to_json(metadata))
    assert metadata_reconstruct == metadata
