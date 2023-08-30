# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import TYPE_CHECKING, List, Tuple, cast, Dict

import matplotlib.textpath

if TYPE_CHECKING:
    import cirq

QBLUE = '#1967d2'
FONT = "Arial"
EMPTY_MOMENT_COLWIDTH = float(21)  # assumed default column width


def fixup_text(text: str):
    if '\n' in text:
        # https://github.com/quantumlib/Cirq/issues/4499
        # TODO: Visualize Custom MatrixGate
        return '?'
    # https://github.com/quantumlib/Cirq/issues/2905
    text = text.replace('[<virtual>]', '')
    text = text.replace('[cirq.VirtualTag()]', '')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    return text


def _get_text_width(t: str) -> float:
    if t == '':  # in case of an empty moment
        return EMPTY_MOMENT_COLWIDTH
    t = fixup_text(t)
    tp = matplotlib.textpath.TextPath((0, 0), t, size=14, prop=FONT)
    bb = tp.get_extents()
    return bb.width + 10


def _rect(
    x: float,
    y: float,
    boxwidth: float,
    boxheight: float,
    fill: str = 'white',
    strokewidth: float = 1,
):
    """Draw an SVG <rect> rectangle."""
    return (
        f'<rect x="{x}" y="{y}" width="{boxwidth}" height="{boxheight}" '
        f'stroke="black" fill="{fill}" stroke-width="{strokewidth}" />'
    )


def _text(x: float, y: float, text: str, fontsize: int = 14):
    """Draw SVG <text> text."""
    return (
        f'<text x="{x}" y="{y}" dominant-baseline="middle" '
        f'text-anchor="middle" font-size="{fontsize}px" '
        f'font-family="{FONT}">{text}</text>'
    )


def _fit_horizontal(
    tdd: 'cirq.TextDiagramDrawer', ref_boxwidth: float, col_padding: float
) -> Tuple[List[float], List[float]]:
    """Figure out the horizontal spacing of columns to fit everything in.

    Returns:
        col_starts: a list of where (in pixels) each column starts.
        col_widths: a list of each column's width in pixels
    """
    max_xi = max(xi for xi, _ in tdd.entries.keys())
    max_xi = max(max_xi, max(cast(int, xi2) for _, _, xi2, _, _ in tdd.horizontal_lines))
    col_widths = [0.0] * (max_xi + 2)
    for (xi, _), v in tdd.entries.items():
        tw = _get_text_width(v.text)
        if tw > col_widths[xi]:
            col_widths[xi] = max(ref_boxwidth, tw)

    for i in range(len(col_widths)):
        # horizontal_padding seems to only zero-out certain paddings
        # we use col_padding as a default
        padding = tdd.horizontal_padding.get(i, col_padding)
        col_widths[i] += padding

    col_starts = [0.0]
    for i in range(1, max_xi + 3):
        col_starts.append(col_starts[i - 1] + col_widths[i - 1])

    return col_starts, col_widths


def _fit_vertical(
    tdd: 'cirq.TextDiagramDrawer', ref_boxheight: float, row_padding: float
) -> Tuple[List[float], List[float], Dict[float, int]]:
    """Return data structures used to turn tdd vertical coordinates into
    well-spaced SVG coordinates.

    The eagle eyed coder may notice that this function is very
    similar to _fit_horizontal. That function was written first
    because horizontal spacing is very important for being able
    to see all the gates but vertical spacing is just for aesthetics.
    It wasn't until this function was written that I (mpharrigan)
    noticed that -- unlike the x-coordinates (which are all integers) --
    y-coordinates come in half-integers. Please use yi_map to convert
    TextDiagramDrawer y-values to y indices which can be used to index
    into row_starts and row_heights.

    See gh-2313 to track this (and other) hacks that could be improved.

    Returns:
        row_starts: A list that maps y indices to the starting y position
            (in SVG px)
        row_heights: A list that maps y indices to the height of each row
            (in SVG px). Y-index `yi` goes from row_starts[yi] to
            row_starts[yi] + row_heights[yi]
        yi_map:
            A mapping from half-integer TextDiagramDrawer coordinates
            to integer y indices. Apply this mapping before indexing into
            the former two return values (ie row_starts and row_heights)
    """
    # Note: y values come as half integers. Map to integers
    all_yis = sorted(
        {yi for _, yi in tdd.entries.keys()}
        | {yi1 for _, yi1, _, _, _ in tdd.vertical_lines}
        | {yi2 for _, _, yi2, _, _ in tdd.vertical_lines}
        | {yi for yi, _, _, _, _ in tdd.horizontal_lines}
    )
    yi_map = {yi: i for i, yi in enumerate(all_yis)}

    max_yi = max(yi_map[yi] for yi in all_yis)
    row_heights = [0.0] * (max_yi + 2)
    for (_, yi), _ in tdd.entries.items():
        yi = yi_map[yi]
        row_heights[yi] = max(ref_boxheight, row_heights[yi])

    for yi_float in all_yis:
        row_heights[yi_map[yi_float]] += row_padding

    row_starts = [0.0]
    for i in range(1, max_yi + 3):
        row_starts.append(row_starts[i - 1] + row_heights[i - 1])

    return row_starts, row_heights, yi_map


def _debug_spacing(col_starts, row_starts):  # pragma: no cover
    """Return a string suitable for inserting inside an <svg> tag that
    draws green lines where columns and rows start. This is very useful
    if you're developing this code and are debugging spacing issues.
    """
    t = ''
    for i, cs in enumerate(col_starts):
        t += (
            f'<line id="cs-{i}" '
            f'x1="{cs}" x2="{cs}" y1="0" y2="{row_starts[-1]}" '
            f'stroke="green" stroke-width="1" />'
        )
    for i, rs in enumerate(row_starts):
        t += (
            f'<line id="rs-{i}" '
            f'x1="0" x2="{col_starts[-1]}" y1="{rs}" y2="{rs}" '
            f'stroke="green" stroke-width="1" />'
        )
    return t


def tdd_to_svg(
    tdd: 'cirq.TextDiagramDrawer',
    ref_boxwidth: float = 40,
    ref_boxheight: float = 40,
    col_padding: float = 20,
    row_padding: float = 10,
) -> str:
    row_starts, row_heights, yi_map = _fit_vertical(
        tdd=tdd, ref_boxheight=ref_boxheight, row_padding=row_padding
    )
    col_starts, col_widths = _fit_horizontal(
        tdd=tdd, ref_boxwidth=ref_boxwidth, col_padding=col_padding
    )

    t = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{col_starts[-1]}" height="{row_starts[-1]}">'
    )

    # Developers: uncomment below to draw green lines to debug
    #             col_starts and row_starts
    # t += _debug_spacing(col_starts, row_starts)

    for yi, xi1, xi2, _, _ in tdd.horizontal_lines:
        xi1 = cast(int, xi1)
        xi2 = cast(int, xi2)
        x1 = col_starts[xi1] + col_widths[xi1] / 2
        x2 = col_starts[xi2] + col_widths[xi2] / 2

        yi = yi_map[yi]
        y = row_starts[yi] + row_heights[yi] / 2

        if xi1 == 0:
            # qubits start at far left and their wires shall be blue
            stroke = QBLUE
        else:
            stroke = 'black'
        t += f'<line x1="{x1}" x2="{x2}" y1="{y}" y2="{y}" stroke="{stroke}" stroke-width="1" />'

    for xi, yi1, yi2, _, _ in tdd.vertical_lines:
        yi1 = yi_map[yi1]
        yi2 = yi_map[yi2]
        y1 = row_starts[yi1] + row_heights[yi1] / 2
        y2 = row_starts[yi2] + row_heights[yi2] / 2

        xi = cast(int, xi)
        x = col_starts[xi] + col_widths[xi] / 2
        t += f'<line x1="{x}" x2="{x}" y1="{y1}" y2="{y2}" stroke="black" stroke-width="3" />'

    for (xi, yi), v in tdd.entries.items():
        yi = yi_map[yi]

        x = col_starts[xi] + col_widths[xi] / 2
        y = row_starts[yi] + row_heights[yi] / 2

        boxheight = ref_boxheight
        boxwidth = col_widths[xi] - tdd.horizontal_padding.get(xi, col_padding)
        boxx = x - boxwidth / 2
        boxy = y - boxheight / 2

        if xi == 0:
            # Qubits
            t += _rect(boxx, boxy, boxwidth, boxheight, strokewidth=0)
            t += _text(x, y, v.text)
            continue

        if v.text == '@':
            t += f'<circle cx="{x}" cy="{y}" r="{ref_boxheight / 4}" />'
            continue
        if v.text == '×':
            t += _text(x, y + 3, '×', fontsize=40)
            continue
        if v.text == '':
            continue

        v_text = fixup_text(v.text)
        t += _rect(boxx, boxy, boxwidth, boxheight)
        t += _text(x, y, v_text, fontsize=14 if len(v_text) > 1 else 18)

    t += '</svg>'
    return t


def _validate_circuit(circuit: 'cirq.Circuit'):
    if len(circuit) == 0:
        raise ValueError("Can't draw SVG diagram for empty circuits")


class SVGCircuit:
    """A wrapper around cirq.Circuit to enable rich display in a Jupyter
    notebook.

    Jupyter will display the result of the last line in a cell. Often,
    this is repr(o) for an object. This class defines a magic method
    which will cause the circuit to be displayed as an SVG image.
    """

    def __init__(self, circuit: 'cirq.Circuit'):
        self.circuit = circuit

    def _repr_svg_(self) -> str:
        return circuit_to_svg(self.circuit)


def circuit_to_svg(circuit: 'cirq.Circuit') -> str:
    """Render a circuit as SVG."""
    _validate_circuit(circuit)
    tdd = circuit.to_text_diagram_drawer(transpose=False)
    if len(tdd.horizontal_lines) == 0:  # in circuits with no non-empty moments,return a blank SVG
        return '<svg></svg>'
    return tdd_to_svg(tdd)
