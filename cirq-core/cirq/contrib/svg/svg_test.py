# pylint: disable=wrong-or-nonexistent-copyright-notice

from __future__ import annotations

import xml.etree.ElementTree as ET

import IPython.display
import numpy as np
import pytest

import cirq
from cirq.contrib.svg import circuit_to_svg, SVGCircuit


def test_svg() -> None:
    a, b, c = cirq.LineQubit.range(3)

    svg_text = circuit_to_svg(
        cirq.Circuit(
            cirq.CNOT(a, b),
            cirq.CZ(b, c),
            cirq.SWAP(a, c),
            cirq.PhasedXPowGate(exponent=0.123, phase_exponent=0.456).on(c),
            cirq.Z(a),
            cirq.measure(a, b, c, key='z'),
            cirq.MatrixGate(np.eye(2)).on(a),
        )
    )
    assert '?' in svg_text
    assert '<svg' in svg_text
    assert '</svg>' in svg_text


def test_svg_noise() -> None:
    noise_model = cirq.ConstantQubitNoiseModel(cirq.DepolarizingChannel(p=1e-3))
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q))
    circuit = cirq.Circuit(noise_model.noisy_moments(circuit.moments, [q]))
    svg = circuit_to_svg(circuit)
    assert '>D(0.001)</text>' in svg


def test_validation() -> None:
    with pytest.raises(ValueError):
        circuit_to_svg(cirq.Circuit())


def test_empty_moments() -> None:
    a, b = cirq.LineQubit.range(2)
    svg_1 = circuit_to_svg(
        cirq.Circuit(
            cirq.Moment(),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(),
            cirq.Moment(cirq.SWAP(a, b)),
            cirq.Moment(cirq.Z(a)),
            cirq.Moment(cirq.measure(a, b, key='z')),
            cirq.Moment(),
        )
    )
    assert '<svg' in svg_1
    assert '</svg>' in svg_1

    svg_2 = circuit_to_svg(cirq.Circuit(cirq.Moment()))
    assert '<svg' in svg_2
    assert '</svg>' in svg_2


@pytest.mark.parametrize(
    'symbol,svg_symbol',
    [
        ('<a', '&lt;a'),
        ('<a&', '&lt;a&amp;'),
        ('<=b', '&lt;=b'),
        ('>c', '&gt;c'),
        ('>=d', '&gt;=d'),
        ('>e<', '&gt;e&lt;'),
        ('A[<virtual>]B[cirq.VirtualTag()]C>D<E', 'ABC&gt;D&lt;E'),
    ],
)
def test_gate_with_less_greater_str(symbol, svg_symbol) -> None:
    class CustomGate(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=[symbol])

    circuit = cirq.Circuit(CustomGate().on(cirq.LineQubit(0)))
    svg_circuit = SVGCircuit(circuit)
    svg = svg_circuit._repr_svg_()

    _ = IPython.display.SVG(svg)
    assert svg_symbol in svg


# A label that, if emitted unescaped into an SVG <text> element, terminates the
# element and injects a <script> node -- i.e. cross-site scripting (XSS) when the
# diagram is rendered inline in a notebook or opened in a browser.
_XSS_PAYLOAD = '</text><script>alert(1)</script>'
_XSS_PAYLOAD_ESCAPED = '&lt;/text&gt;&lt;script&gt;alert(1)&lt;/script&gt;'


def _assert_no_markup_injection(svg: str) -> None:
    """Asserts `svg` is well-formed XML with no injected markup."""
    # No raw markup broke out of a <text> element.
    assert '<script>' not in svg
    assert '</text><script>' not in svg
    # The output is still well-formed XML (escaping did not corrupt it). A
    # successful parse here is the actual proof the payload stayed inert data.
    ET.fromstring(svg)
    # IPython display wrapper should also accept it.
    _ = IPython.display.SVG(svg)


@pytest.mark.parametrize(
    'name,escaped',
    [
        ('<a', '&lt;a'),
        ('a&b', 'a&amp;b'),
        ('>c<', '&gt;c&lt;'),
        (_XSS_PAYLOAD, _XSS_PAYLOAD_ESCAPED),
    ],
)
def test_qubit_label_is_escaped(name, escaped) -> None:
    # Attack surface 1: qubit names. These are user/attacker controlled (e.g. via
    # a circuit loaded with `cirq.read_json`) and are rendered as the content of
    # the left-most SVG <text> element.
    q = cirq.NamedQubit(name)
    svg = SVGCircuit(cirq.Circuit(cirq.X(q)))._repr_svg_()

    _assert_no_markup_injection(svg)
    assert escaped in svg


@pytest.mark.parametrize(
    'symbol,escaped',
    [
        ('<a', '&lt;a'),
        ('a&b', 'a&amp;b'),
        ('>c<', '&gt;c&lt;'),
        (_XSS_PAYLOAD, _XSS_PAYLOAD_ESCAPED),
    ],
)
def test_gate_label_is_escaped(symbol, escaped) -> None:
    # Attack surface 2: gate wire symbols. A custom (or deserialized) gate can
    # expose an arbitrary diagram symbol, which is rendered as <text> content too.
    # Both surfaces funnel through the single `_text` sink, so both are protected.
    class EvilGate(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=[symbol])

    svg = SVGCircuit(cirq.Circuit(EvilGate().on(cirq.LineQubit(0))))._repr_svg_()

    _assert_no_markup_injection(svg)
    assert escaped in svg


def test_normal_rendering_is_unchanged() -> None:
    # Escaping must not alter rendering of ordinary, safe labels: legitimate gate
    # and qubit text is emitted verbatim and the document remains valid XML.
    a, b = cirq.LineQubit.range(2)
    svg = circuit_to_svg(cirq.Circuit(cirq.H(a), cirq.X(b), cirq.CNOT(a, b)))

    assert '>H</text>' in svg
    assert '>X</text>' in svg
    assert '>0: </text>' in svg  # qubit label rendered normally (verbatim)
    ET.fromstring(svg)


# Adversarial label payloads spanning several markup-injection techniques: raw
# script/img/svg elements, breaking out of the enclosing <text> node, attribute
# breakout, CDATA/comment/entity tricks, pre-escaped and numeric entities, and a
# large input. After contextual escaping, *none* may introduce new markup.
_FUZZ_PAYLOADS = [
    '<script>alert(1)</script>',
    '</text><script>alert(1)</script>',
    '"><script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<svg onload=alert(1)>',
    ']]><!--',
    '<!ENTITY xxe SYSTEM "file:///etc/passwd">',
    '&amp;&lt;&gt;',  # already-escaped text must be re-escaped, never interpreted
    '&#60;script&#62;',  # numeric character references
    '<<>>&&',
    '<' * 120,  # moderately large markup-heavy input
]

# The complete set of element names the SVG serializer is ever allowed to emit.
# If a label could inject markup, a parse of the output would reveal a tag outside
# this set -- so asserting the tag set is a subset of this is a direct proof that
# the entire class of markup-injection (XSS) is eliminated, not just one payload.
_ALLOWED_SVG_TAGS = {'svg', 'rect', 'text', 'line', 'circle'}


def _circuit_with_label(payload: str, surface: str) -> cirq.Circuit:
    if surface == 'qubit':
        return cirq.Circuit(cirq.X(cirq.NamedQubit(payload)))

    class _LabeledGate(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=[payload])

    return cirq.Circuit(_LabeledGate().on(cirq.LineQubit(0)))


@pytest.mark.parametrize('surface', ['qubit', 'gate'])
@pytest.mark.parametrize('payload', _FUZZ_PAYLOADS)
def test_svg_renderer_neutralizes_label_injection(payload, surface) -> None:
    # Fuzz both attacker-controlled label surfaces with a battery of payloads and
    # assert the structural invariant: the output parses as XML and contains only
    # the fixed SVG element vocabulary, i.e. no label can ever become markup.
    svg = SVGCircuit(_circuit_with_label(payload, surface))._repr_svg_()

    root = ET.fromstring(svg)  # must remain well-formed XML
    tags = {elem.tag.split('}')[-1] for elem in root.iter()}  # strip {namespace}
    assert tags <= _ALLOWED_SVG_TAGS, f'injected element(s): {tags - _ALLOWED_SVG_TAGS}'

    # Belt-and-suspenders: no raw dangerous start-tag survived as markup.
    assert '<script' not in svg
    assert '<img' not in svg


@pytest.mark.parametrize('surface', ['qubit', 'gate'])
@pytest.mark.parametrize('payload', ['(((((', '{{{{{', '====', 'pi/2', 'a\nb', '00000', '-3.14'])
def test_svg_renderer_robust_to_malformed_labels(payload, surface) -> None:
    # Robustness: odd-but-benign labels must never crash the serializer and must
    # always yield well-formed XML.
    svg = SVGCircuit(_circuit_with_label(payload, surface))._repr_svg_()
    ET.fromstring(svg)
