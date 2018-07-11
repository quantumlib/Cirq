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


from typing import Optional, Type, TypeVar, TYPE_CHECKING, Generic

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq.extension.extensions

T_DESIRED = TypeVar('T_DESIRED')


T_AVAILABLE_UNION = TypeVar('T_AVAILABLE_UNION')


class PotentialImplementation(Generic[T_AVAILABLE_UNION]):
    """For values that may or may not implement an interface / feature, but can
    determine if so at runtime.

    Extensions uses this as a fallback when trying to cast to a desired type.

    TypeArgs:
        T_AVAILABLE_UNION: A type or union of types that the child class may
            implement. At the moment this is purely for documentation, with no
            effect at runtime or when checking types.
            TODO: use variadic type ( but github.com/python/typing/issues/193 )
    """

    def try_cast_to(self,
                    desired_type: Type[T_DESIRED],
                    extensions: 'cirq.extension.extensions.Extensions'
                    ) -> Optional[T_DESIRED]:
        """Turns this value into the desired type, if possible.

        Correct implementations should delegate to super() after failing to
        cast, instead of returning None.

        Args:
            desired_type: The type of thing that the caller wants to use.
            extensions: The extensions instance that is asking us to try to
                cast ourselves into something as part of its try_cast method.
                If we need to recursively cast some of our fields in order to
                cast ourselves, this is the extensions instance we should use.

        Returns:
            None if the receiving instance doesn't recognize or can't implement
                the desired type. Otherwise a value that meets the interface.
        """
        if isinstance(self, desired_type):
            return self
        return None
