"""Package containing code for interoperating with other quantum software."""

from cirq.interop.quirk import (
    quirk_json_to_circuit,
    quirk_url_to_circuit,
)

from cirq.interop.qasm.qasm_output import QasmOutput