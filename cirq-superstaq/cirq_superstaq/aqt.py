import importlib
from typing import List, Optional, Union

import applications_superstaq
import cirq

import cirq_superstaq

try:
    import qtrl.sequencer
except ModuleNotFoundError:
    pass


class AQTCompilerOutput:
    def __init__(
        self,
        circuits: Union[cirq.Circuit, List[cirq.Circuit]],
        seq: Optional["qtrl.sequencer.Sequence"] = None,
        pulse_lists: Optional[Union[List[List], List[List[List]]]] = None,
    ) -> None:
        if isinstance(circuits, cirq.Circuit):
            self.circuit = circuits
            self.pulse_list = pulse_lists
        else:
            self.circuits = circuits
            self.pulse_lists = pulse_lists

        self.seq = seq

    def has_multiple_circuits(self) -> bool:
        """Returns True if this object represents multiple circuits.

        If so, this object has .circuits and .pulse_lists attributes. Otherwise, this object
        represents a single circuit, and has .circuit and .pulse_list attributes.
        """
        return hasattr(self, "circuits")

    def __repr__(self) -> str:
        if not self.has_multiple_circuits():
            return f"AQTCompilerOutput({self.circuit!r}, {self.seq!r}, {self.pulse_list!r})"
        return f"AQTCompilerOutput({self.circuits!r}, {self.seq!r}, {self.pulse_lists!r})"


def read_json(json_dict: dict, circuits_list: bool) -> AQTCompilerOutput:
    """Reads out returned JSON from SuperstaQ API's AQT compilation endpoint.

    Args:
        json_dict: a JSON dictionary matching the format returned by /aqt_compile endpoint
        circuits_list: bool flag that controls whether the returned object has a .circuits
            attribute (if True) or a .circuit attribute (False)
    Returns:
        a AQTCompilerOutput object with the compiled circuit(s). If qtrl is available locally,
        the returned object also stores the pulse sequence in the .seq attribute and the
        list(s) of cycles in the .pulse_list(s) attribute.
    """
    seq = None
    pulse_lists = None

    if importlib.util.find_spec(
        "qtrl"
    ):  # pragma: no cover, b/c qtrl is not open source so it is not in cirq-superstaq reqs
        state = applications_superstaq.converters.deserialize(json_dict["state_jp"])

        seq = qtrl.sequencer.Sequence(n_elements=1)
        seq.__setstate__(state)
        seq.compile()

        pulse_lists = applications_superstaq.converters.deserialize(json_dict["pulse_lists_jp"])

    resolvers = [cirq_superstaq.custom_gates.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    compiled_circuits = [
        cirq.read_json(json_text=c, resolvers=resolvers) for c in json_dict["cirq_circuits"]
    ]
    if circuits_list:
        return AQTCompilerOutput(compiled_circuits, seq, pulse_lists)

    pulse_list = pulse_lists[0] if pulse_lists is not None else None
    return AQTCompilerOutput(compiled_circuits[0], seq, pulse_list)
