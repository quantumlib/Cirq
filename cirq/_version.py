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

"""Define version number here, read it from setup.py automatically,
and warn users that the latest version of cirq uses python 3.6+"""

import sys

if sys.version_info < (3, 6, 0):
    # coverage: ignore
    raise SystemError(
        "You installed the latest version of cirq but aren't on python 3.6+.\n"
        'To fix this error, you need to either:\n'
        '\n'
        'A) Update to python 3.6 or later.\n'
        '- OR -\n'
        'B) Explicitly install an older deprecated-but-compatible version '
        'of cirq (e.g. "python -m pip install cirq==0.5.*")')

__version__ = "0.10.0.dev"
