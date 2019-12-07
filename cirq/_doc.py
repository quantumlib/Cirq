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
"""Workaround for associating docstrings with public constants."""
from typing import Any, Dict, NamedTuple, Optional

DocProperties = NamedTuple(
    'DocProperties',
    [
        ('doc_string', Optional[str]),
    ],
)

RECORDED_CONST_DOCS: Dict[int, DocProperties] = {}


def document(value: Any, doc_string: str):
    """Stores documentation details about the given value.

    This method is used to associate a docstring with global constants.

    The given documentation information is filed under `id(value)` in
    `cirq._doc.RECORDED_CONST_DOCS`.

    Args:
        value: The value to associate with documentation information.
        doc_string: The doc string to associate with the value.
    """
    docs = DocProperties(doc_string=doc_string)
    RECORDED_CONST_DOCS[id(value)] = docs
