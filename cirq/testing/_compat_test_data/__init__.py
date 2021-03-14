"""Test module for deprecation testing.
See cirq/_compat_test.py for the tests.
This module contains example deprecations for modules.
"""

from logging import info
from cirq import _compat

info("init:compat_test_data")

# simulates a rename of this module
# fake_a -> module_a
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.module_a",
    old_parent=__name__,
    old_child="fake_a",
    deadline="v0.20",
)

# simulates a move of this module
# fake_b -> module_a.module_b
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.module_a.module_b",
    old_parent=__name__,
    old_child="fake_b",
    deadline="v0.20",
)

# simulates a move of this module
# fake_b -> module_a.module_b
_compat.deprecated_submodule(
    new_module_name="cirq.google",
    old_parent=__name__,
    old_child="fake_google",
    deadline="v0.20",
)
