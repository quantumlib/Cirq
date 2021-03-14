from logging import info

from cirq import _compat

info("init:compat_test_data")

_compat.deprecated_submodule(
    new_module_name="cirq._compat_test_data.module_a",
    old_parent=__name__,
    old_child="fake_a",
    deadline="v0.20",
)

_compat.deprecated_submodule(
    new_module_name="cirq._compat_test_data.module_a.module_b",
    old_parent=__name__,
    old_child="fake_b",
    deadline="v0.20",
)

_compat.deprecated_submodule(
    new_module_name="cirq.google",
    old_parent=__name__,
    old_child="fake_google",
    deadline="v0.20",
)
