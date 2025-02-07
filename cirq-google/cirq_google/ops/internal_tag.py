# Copyright 2025 The Cirq Developers
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

from typing import Any, Dict

from cirq import value


@value.value_equality
class InternalTag:
    """InternalTag is a placeholder tag for internal tags that
    are not specified publicly.

    These tags can hold the information needed to instantiate a
    tag specified in an internal library.
    """

    def __init__(self, name: str, package: str, **kwargs):
        """Instantiates an InternalTag.

        Arguments:
            name: Tag class name.
            package: The python module of the tag.
            **kwargs: Arbitrary keyword parameters that should be passed to the tag.
        """
        self.name = name
        self.package = package
        self.tag_args = kwargs

    def __str__(self):
        tag_args = ', '.join(f'{k}={v}' for k, v in self.tag_args.items())
        return f'{self.package}.{self.name}({tag_args})'

    def __repr__(self) -> str:
        tag_args = ', '.join(f'{k}={v!r}' for k, v in self.tag_args.items())
        if tag_args:
            tag_args = ', ' + tag_args
        return f"cirq_google.InternalTag(name='{self.name}', package='{self.package}'{tag_args})"

    def _json_dict_(self) -> Dict[str, Any]:
        return dict(name=self.name, package=self.package, **self.tag_args)

    def _value_equality_values_(self):
        try:
            tag_args_eq_values = frozenset(self.tag_args.items())
        except TypeError:
            tag_args_eq_values = self.tag_args
        return (self.name, self.package, tag_args_eq_values)
