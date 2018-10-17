# Copyright 2018 The Cirq Developers
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

from typing import Any

# At the moment there's no reliable way to say 'NotImplementedType'.
# - There is a NotImplementedType in python 2, but not python 3.
# - type(NotImplemented) causes mypy to error.
# - Just NotImplemented causes runtime errors (it's not a type).
# - The string "NotImplemented" causes runtime errors (in some python versions).
NotImplementedType = Any
