# Copyright 2018 Google LLC
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


from typing import Optional, Type, TypeVar

T_DESIRED = TypeVar('TDesired')


class PotentialImplementation:
    """For values that may or may not implement an interface / feature, but can
    determine if so at runtime.

    Extensions uses this as a fallback when trying to cast to a desired type.
    """

    def try_cast_to(self, desired_type: Type[T_DESIRED]
                    ) -> Optional[T_DESIRED]:
        """Turns this value into the desired type, if possible.

        Correct implementations should delegate to super() after failing to
        cast, instead of returning None.

        Args:
            desired_type: The type of thing that the caller wants to use.

        Returns:
            None if the receiving instance doesn't recognize or can't implement
                the desired type. Otherwise a value that meets the interface.
        """
        if isinstance(self, desired_type):
            return self
        return None
