from cirq.ops import raw_types


class LineQubit(raw_types.QubitId):
    """A qubit on a 1d lattice with nearest-neighbor connectivity."""

    def __init__(self, x: int) -> None:
        """Initializes a line qubit at the given x coordinate."""
        self.x = x

    def is_adjacent(self, other: 'LineQubit') -> bool:
        """Determines if two line qubits are adjacent."""
        return abs(self.x - other.x) == 1

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.x == other.x

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((LineQubit, self.x))

    def __repr__(self):
        return 'LineQubit({})'.format(self.x)

    def __str__(self):
        return '{}'.format(self.x)
