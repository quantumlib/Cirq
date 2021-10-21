# pylint: disable=wrong-or-nonexistent-copyright-notice
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

# simulates a rename of a child module with same prefix
# this prefix will collide with multiple "old" and new prefixes here
# module_ -> module_a
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.module_a",
    old_parent=__name__,
    old_child="module_",
    deadline="v0.20",
    create_attribute=False,
)


# simulates a move of sub module to one below
# fake_b -> module_a.module_b
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.module_a.module_b",
    old_parent=__name__,
    old_child="fake_b",
    deadline="v0.20",
    create_attribute=False,
)

# simulates a move of fake_ops child module to a different parent
# fake_ops -> cirq.ops
_compat.deprecated_submodule(
    new_module_name="cirq.ops",
    old_parent=__name__,
    old_child="fake_ops",
    deadline="v0.20",
    create_attribute=False,
)


# simulates a move of child module to a top level module.
# this will be the case with cirq.google -> cirq_google
# fake_freezegun -> freezegun
_compat.deprecated_submodule(
    new_module_name="freezegun",
    old_parent=__name__,
    old_child="fake_freezegun",
    deadline="v0.20",
    create_attribute=False,
)

# simulates a move of this module
# fake_c -> module_a.module_b.module_c
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.module_a.module_b.module_c",
    old_parent=__name__,
    old_child="fake_c",
    deadline="v0.20",
    create_attribute=False,
)


# this is admittedly contrived
# simulates a move of fake_child to a repeated substructure
# cirq.testing._compat_test_data.repeated_child ->
# cirq.testing._compat_test_data.repeated
# now...repeated has also a submodule named cirq.testing._compat_test_data.repeated_child.child
# which adds to
# cirq.testing._compat_test_data.repeated_child.cirq.testing._compat_test_data.repeated_child.child
_compat.deprecated_submodule(
    new_module_name=f"{__name__}.repeated",
    old_parent=__name__,
    old_child="repeated_child",
    deadline="v0.20",
    create_attribute=False,
)

# a missing module that is setup as a broken reference
_compat.deprecated_submodule(
    new_module_name='missing_module',
    old_parent=__name__,
    old_child='broken_ref',
    deadline='v0.20',
    create_attribute=True,
)
