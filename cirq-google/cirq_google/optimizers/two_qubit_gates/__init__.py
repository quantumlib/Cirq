# pylint: disable=wrong-or-nonexistent-copyright-notice
from cirq import _compat

from cirq_google.optimizers.two_qubit_gates.gate_compilation import (
    gate_product_tabulation,
    GateTabulation,
)

_compat.deprecated_submodule(
    new_module_name="cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils",
    old_parent=__name__,
    old_child="math_utils",
    deadline="v0.16",
    create_attribute=True,
)
