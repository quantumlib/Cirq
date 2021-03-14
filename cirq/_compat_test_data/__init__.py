from logging import info

from cirq import _compat

info("TEST DATA INIT ")
_compat.deprecated_submodule(
    new_module_name="cirq.google.calibration",
    old_parent="cirq._compat_test_data",
    old_child="fake_calibration",
    deadline="v0.20",
)
