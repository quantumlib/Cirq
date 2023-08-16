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
import functools

from qcs_api_client.client import build_sync_client


def _provide_default_client(function):
    """A decorator that will initialize an `httpx.Client` and pass
    it to the wrapped function as a kwarg if not already present. This
    eases provision of a default `httpx.Client` with Rigetti
    QCS configuration and authentication. If the decorator initializes a
    default client, it will invoke the wrapped function from within the
    `httpx.Client` context.

    Args:
        function: The decorated function.

    Returns:
        The `function` wrapped with a default `client`.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if 'client' in kwargs:
            return function(*args, **kwargs)

        with build_sync_client() as client:  # pragma: no cover
            kwargs['client'] = client
            return function(*args, **kwargs)

    return wrapper
