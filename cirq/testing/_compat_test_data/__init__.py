"""Test module for deprecation testing.
See cirq/_compat_test.py for the tests.
This module contains example deprecations for modules.
"""
import sys
from logging import info
from cirq import _compat

info("init:compat_test_data")

# simulates a rename of a child module
# fake_a -> module_a
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.module_a",
    old_parent=__name__,
    old_child="fake_a",
    deadline="v0.20",
    create_attribute=True,
)

# simulates a move of sub module to one below
# fake_b -> module_a.module_b
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.module_a.module_b",
    old_parent=__name__,
    old_child="fake_b",
    deadline="v0.20",
    create_attribute=True,
)

# simulates a move of fake_ops child module to a different parent
# fake_ops -> cirq.ops
_compat.deprecated_submodule(
    new_module_name="cirq.ops",
    old_parent=__name__,
    old_child="fake_ops",
    deadline="v0.20",
    create_attribute=True,
)


# simulates a move of child module to a top level module.
# this will be the case with cirq.google -> cirq_google
# fake_freezegun -> freezegun
_compat.deprecated_submodule(
    new_module_name="freezegun",
    old_parent=__name__,
    old_child="fake_freezegun",
    deadline="v0.20",
    create_attribute=True,
)

# simulates a move of this module
# fake_c -> module_a.module_b.module_c
# but this won't create an attribute!
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.module_a.module_b.module_c",
    old_parent=__name__,
    old_child="fake_c",
    deadline="v0.20",
    create_attribute=True,
)
