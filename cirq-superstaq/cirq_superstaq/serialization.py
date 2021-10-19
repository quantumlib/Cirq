from typing import List, Union

import cirq

import cirq_superstaq


def serialize_circuits(circuits: Union[cirq.Circuit, List[cirq.Circuit]]) -> str:
    """Serialize Circuit(s) into a json string

    Args:
        circuits: a Circuit or list of Circuits to be serialized

    Returns:
        str representing the serialized circuit(s)
    """
    return cirq.to_json(circuits)


def deserialize_circuits(serialized_circuits: str) -> Union[cirq.Circuit, List[cirq.Circuit]]:
    """Deserialize serialized Circuit(s)

    Args:
        serialized_circuits: json str generated via converters.serialize_circuit()

    Returns:
        the Circuit or list of Circuits that was serialized
    """
    resolvers = [cirq_superstaq.custom_gates.custom_resolver, *cirq.DEFAULT_RESOLVERS]
    return cirq.read_json(json_text=serialized_circuits, resolvers=resolvers)
