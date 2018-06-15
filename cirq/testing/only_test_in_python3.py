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

import sys


def only_test_in_python3(func):
    """A decorator that indicates a test should not execute in python 2.

    For example, in python 2 repr('a') is "u'a'" instead of "'a'" when
    from __future__ import unicode is present (which it will be, since 3to2
    inserts it for us). This is annoying to work around when testing repr
    methods, so instead you can just tag the test with this decorator.
    """
    if sys.version_info.major < 3:
        return None  # coverage: ignore
    return func
