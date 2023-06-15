# Copyright 2021 The Cirq Developers
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


def pytest_configure(config):
    # Ignore deprecation warnings in python code generated from our protobuf definitions.
    # Eventually, the warnings will be removed by upgrading protoc compiler. See issues
    # #4161 and #4737.
    for f in (
        "FieldDescriptor",
        "Descriptor",
        "EnumDescriptor",
        "EnumValueDescriptor",
        "FileDescriptor",
        "OneofDescriptor",
    ):
        config.addinivalue_line("filterwarnings", f"ignore:Call to deprecated create function {f}")


def pytest_addoption(parser):
    parser.addoption(
        "--rigetti-integration",
        action="store_true",
        default=False,
        help="run Rigetti integration tests",
    )
