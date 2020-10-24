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

from typing import Any, Dict

RECORDED_CONST_DOCS: Dict[int, str] = {}


def document(value: Any, doc_string: str = ''):
    """Stores documentation details about the given value.

    This method is used to associate a docstring with global constants. It is
    also used to indicate that a private method should be included in the public
    documentation (e.g. when documenting protocols or arithmetic operations).

    The given documentation information is filed under `id(value)` in
    `cirq._doc.RECORDED_CONST_DOCS`.

    Args:
        value: The value to associate with documentation information.
        doc_string: The doc string to associate with the value. Defaults to the
            value's __doc__ attribute.

    Returns:
        The given value.
    """
    RECORDED_CONST_DOCS[id(value)] = doc_string

    ## this is how the devsite API generator picks up
    ## docstrings for type aliases
    try:
        value.__doc__ = doc_string
    except AttributeError:
        # we have a couple (~ 7) global constants of type list, tuple, dict,
        # that fail here as their __doc__ attribute is read-only.
        # For the time being these are not going to be part of the generated
        # API docs. See https://github.com/quantumlib/Cirq/issues/3276 for
        # more info.

        # to list them, uncomment these lines and run `import cirq`:
        # print(f"WARNING: {e}")
        # print(traceback.format_stack(limit=2)[0])
        pass
    return value


# This is based on
# https://github.com/tensorflow/docs/commit/129e54b1a1dc2c2c82ad94bc81e986c7c2be3d6a#diff-85111596b523b2940651a8856939755c8531d470948895c7133deb6a537bc889R295-R324

_DOC_PRIVATE = "_tf_docs_doc_private"


def doc_private(obj):
    """A decorator: Generates docs for private methods/functions.

    For example:
    ```
    class Try:
      @doc_private
      def _private(self):
        ...
    ```
    As a rule of thumb, private (beginning with `_`) methods/functions are
    not documented. This decorator allows to force document a private
    method/function.

    Args:
      obj: The class-attribute to force the documentation for.
    Returns:
      obj
    """

    setattr(obj, _DOC_PRIVATE, None)
    return obj
