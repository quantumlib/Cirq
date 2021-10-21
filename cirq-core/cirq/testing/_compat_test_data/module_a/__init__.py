# pylint: disable=wrong-or-nonexistent-copyright-notice
"""module_a for module deprecation tests"""

from logging import info

from cirq.testing._compat_test_data.module_a import (
    module_b,
)

from cirq.testing._compat_test_data.module_a.dupe import (
    DUPE_CONSTANT,
)

from cirq.testing._compat_test_data.module_a.types import (
    SampleType,
)

MODULE_A_ATTRIBUTE = "module_a"

info("init:module_a")
