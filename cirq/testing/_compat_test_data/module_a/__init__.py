from logging import info

from cirq.testing._compat_test_data.module_a import (
    module_b,
)

from cirq.testing._compat_test_data.module_a.dupe import (
    DUPE_CONSTANT,
)

info("init:module_a")
