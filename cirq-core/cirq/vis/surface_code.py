import enum
import itertools
from typing import Iterator, Set, Tuple

import matplotlib.patches
import matplotlib.pyplot as plt
import networkx as nx
import dataclasses

Coord = Tuple[int, int]


def _meas_with_boundaries(width, height) -> Iterator[Coord]:
    """Helper iterator to yield the correct boundary measure qubits for a `width, height` patch."""
    for x, y in itertools.product(range(width + 1), range(height + 1)):
        if x == 0 and y % 2 == 1:
            continue
        if y == 0 and x % 2 == 0:
            continue
        if x == width and y % 2 == 0:
            continue
        if y == height and x % 2 == 1:
            continue

        yield x, y


def get_data_and_meas_qubits(
    width: int, height: int, offset_x: int = 0, offset_y: int = 0
) -> Tuple[Set[Coord], Set[Coord]]:
    """Return (x,y) coordinates for data and measure qubits.

    Inputs use the "data qubit coordinate system" where width=height=d produces
    enough qubits to support a distance-d surface code.

    Output coordinates use the "all qubits coordinate system" where data qubit (i,j) is
    at position (2i, 2j) and measure qubits are at (2i-1, 2j-1). All coordinates in the
    "all qubits coordinate system" are integers.

    Returns:
        Two sets containing the data and measure qubit coordinates, respectively.
    """
    data = set(
        (2 * x + 2 * offset_x, 2 * y + 2 * offset_y) for x in range(width) for y in range(height)
    )
    meas = set(
        (2 * x - 1 + 2 * offset_x, 2 * y - 1 + 2 * offset_y)
        for x, y in _meas_with_boundaries(width, height)
    )
    return data, meas


def wire_up_graph(data: Set[Coord], meas: Set[Coord]) -> nx.Graph:
    """Given `data` and `meas` qubits, return a graph.

    Neighboring qubits are connected according to the surface code layout.
    Consider using `get_data_and_meas_qubits` to get the inputs to this function.
    """
    nodes = data | meas
    g = nx.Graph()
    for node in nodes:
        for dx, dy in [(1, -1), (1, 1), (-1, -1), (-1, 1)]:
            neighbor = (node[0] + dx, node[1] + dy)
            if neighbor in nodes:
                g.add_edge(node, neighbor)

    return g


class StabShape(enum.Enum):
    """Stabilizer shape.

    Can be `BULK` for a good-ol' square or LEFT, RIGHT, TOP, BOT corresponding
    to weight-two stabalizers on the respective boundary.
    """

    BULK = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()
    TOP = enum.auto()
    BOT = enum.auto()


@dataclasses.dataclass(frozen=True)
class Stabilizer:
    """A stabilizer coordinates, type (X or Z), and shape."""

    x: int
    y: int
    stabtype: bool
    shape: StabShape = StabShape.BULK


def iter_boundary_stabilizers(
    width: int, height: int, orientation: bool, offset_x: int = 0, offset_y: int = 0
) -> Iterator[Stabilizer]:
    """Yield the left, right, top, and bottom weight-2 stabilizers."""
    x = -1
    for y in range(1, height, 2):
        yield Stabilizer(
            x + offset_x, y + offset_y, y % 2 == int(not orientation), shape=StabShape.LEFT
        )

    y = -1
    for x in range(0, width - 1, 2):
        yield Stabilizer(
            x + offset_x, y + offset_y, x % 2 == int(not orientation), shape=StabShape.BOT
        )

    x = width - 1
    for y in range(0, height - 1, 2):
        yield Stabilizer(
            x + offset_x, y + offset_y, y % 2 == int(orientation), shape=StabShape.RIGHT
        )

    y = height - 1
    for x in range(1, width, 2):
        yield Stabilizer(x + offset_x, y + offset_y, x % 2 == int(orientation), shape=StabShape.TOP)


def iter_stabilizers(
    width: int, height: int, orientation: bool, offset_x: int = 0, offset_y: int = 0
) -> Iterator[Stabilizer]:
    """Yield all the stabilizers for a tile of the given dimensions, orientation, and position."""
    for x in range(width - 1):
        for y in range(height - 1):
            fc = (x + y) % 2 == int(orientation)
            yield Stabilizer(x + offset_x, y + offset_y, fc)

    yield from iter_boundary_stabilizers(width, height, orientation, offset_x, offset_y)


def _semi_c(x, y, theta, **kwargs):
    """Return a semi-circle patch.

    `(x,y)` is the center (*not* bottom left). `theta` is where the wedge starts (in degrees).
    """
    return matplotlib.patches.Wedge((x, y), 1, theta, theta + 180, **kwargs)


STABSTYLE = dict(zorder=20, alpha=0.2)
STABPATCH = {
    StabShape.BULK: lambda x, y, fc: plt.Rectangle((2 * x, 2 * y), 2, 2, fc=fc, **STABSTYLE),
    StabShape.LEFT: lambda x, y, fc: _semi_c(2 * x + 2, 2 * y + 1, 90, fc=fc, **STABSTYLE),
    StabShape.BOT: lambda x, y, fc: _semi_c(2 * x + 1, 2 * y + 2, 180, fc=fc, **STABSTYLE),
    StabShape.RIGHT: lambda x, y, fc: _semi_c(2 * x, 2 * y + 1, -90, fc=fc, **STABSTYLE),
    StabShape.TOP: lambda x, y, fc: _semi_c(2 * x + 1, 2 * y, 0, fc=fc, **STABSTYLE),
}
"""Patches for each of the stabilizer shapes (so we can draw nice semi-circles)."""


def draw_tile(
    width: int,
    height: int,
    orientation: bool,
    offset_x: int = 0,
    offset_y: int = 0,
    *,
    ax: plt.Axes = None,
    qubit_size: int = 5,
    draw_couplings: bool = True,
    draw_stabilizers: bool = True,
):
    """Draw a surface code tile.

    Args:
        width, height: The width and height (in data-qubit units) of the tile.
        orientation: which stabilizer type will be top/bottom. `True` draws red
            boundaries on the top and bottom.
        offset_(x,y): Where to start drawing (in data-qubit units) the tile.
        ax: The matplotlib axes to draw on.
        qubit_size: The point size used to draw individual qubits. Set to `0` to omit the
            drawing of qubits.
        draw_couplings: Whether to draw couplings (edges) between qubits.
        draw_stabilizers: Whether to draw the stabilizers.
    """
    if ax is None:
        ax = plt.gca()

    data, meas = get_data_and_meas_qubits(width, height, offset_x, offset_y)

    if qubit_size > 0:
        ax.scatter(
            [x for x, y in data], [y for x, y in data], zorder=10, color='black', s=qubit_size**2
        )
        ax.scatter(
            [x for x, y in meas],
            [y for x, y in meas],
            zorder=10,
            color='white',
            ec='black',
            s=qubit_size**2,
        )

    if draw_couplings:
        g = wire_up_graph(data, meas)
        nx.draw_networkx_edges(g, pos={n: n for n in g.nodes}, ax=ax)

    if draw_stabilizers:
        for stab in iter_stabilizers(width, height, orientation, offset_x, offset_y):
            fc = 'red' if stab.stabtype else 'blue'
            ax.add_patch(STABPATCH[stab.shape](stab.x, stab.y, fc))

    ax.axis('equal')


def draw_tile_outline(
    width: int,
    height: int,
    orientation: bool,
    offset_x: int = 0,
    offset_y: int = 0,
    *,
    ax: plt.Axes = None,
):
    """Draw the outline and boundaries of a surface code tile.

    Args:
        width, height: The width and height (in data-qubit units) of the tile.
        orientation: which stabilizer type will be top/bottom. `True` draws red
            boundaries on the top and bottom.
        offset_(x,y): Where to start drawing (in data-qubit units) the tile.
    """
    if ax is None:
        ax = plt.gca()

    # Outline
    ax.add_patch(
        plt.Rectangle(
            (2 * offset_x, 2 * offset_y), width * 2 - 2, height * 2 - 2, fc='none', ec='black'
        )
    )

    # Boundaries
    for stab in iter_boundary_stabilizers(width, height, orientation, offset_x, offset_y):
        fc = 'red' if stab.stabtype else 'blue'
        patch = STABPATCH[stab.shape](stab.x, stab.y, fc)
        patch.set_alpha(1.0)  # make the boundaries easier to see.
        ax.add_patch(patch)

    ax.axis('equal')
