<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.interop.quirk_url_to_circuit" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.interop.quirk_url_to_circuit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/interop/quirk/url_to_circuit.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Parses a Cirq circuit out of a Quirk URL.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.interop.quirk.quirk_url_to_circuit`, `cirq.interop.quirk.url_to_circuit.quirk_url_to_circuit`, `cirq.quirk_url_to_circuit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.interop.quirk_url_to_circuit(
    quirk_url: str,
    *,
    qubits: Optional[Sequence['cirq.Qid']] = None,
    extra_cell_makers: Union[Dict[str, 'cirq.Gate'], Iterable['cirq.interop.quirk.cells.CellMaker']] = (),
    max_operation_count: int = (10 ** 6)
) -> "cirq.Circuit"
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`quirk_url`
</td>
<td>
The URL of a bookmarked Quirk circuit. It is not required
that the domain be "algassert.com/quirk". The only important part of
the URL is the fragment (the part after the #).
</td>
</tr><tr>
<td>
`qubits`
</td>
<td>
Qubits to use in the circuit. The length of the list must be
at least the number of qubits in the Quirk circuit (including unused
qubits). The maximum number of qubits in a Quirk circuit is 16.
This argument defaults to <a href="../../cirq/devices/LineQubit.md#range"><code>cirq.LineQubit.range(16)</code></a> when not
specified.
</td>
</tr><tr>
<td>
`extra_cell_makers`
</td>
<td>
Non-standard Quirk cells to accept. This can be
used to parse URLs that come from a modified version of Quirk that
includes gates that Quirk doesn't define. This can be specified
as either a list of <a href="../../cirq/interop/quirk/cells/CellMaker.md"><code>cirq.interop.quirk.cells.CellMaker</code></a> instances,
or for more simple cases as a dictionary from a Quirk id string
to a cirq Gate.
</td>
</tr><tr>
<td>
`max_operation_count`
</td>
<td>
If the number of operations in the circuit would
exceed this value, the method raises a `ValueError` instead of
attempting to construct the circuit. This is important to specify
for servers parsing unknown input, because Quirk's format allows for
a billion laughs attack in the form of nested custom gates.
</td>
</tr>
</table>



#### Examples:

>>> print(cirq.quirk_url_to_circuit(
...     'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}'
... ))

* <b>`0`</b>: ───H───@───
          │
* <b>`1`</b>: ───────X───

```
>>> print(cirq.quirk_url_to_circuit(
...     'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}',
...     qubits=[cirq.NamedQubit('Alice'), cirq.NamedQubit('Bob')]
... ))
* <b>`Alice`</b>: ───H───@───
              │
* <b>`Bob`</b>: ─────────X───
```

```
>>> print(cirq.quirk_url_to_circuit(
...     'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
...     extra_cell_makers={'iswap': cirq.ISWAP}))
* <b>`0`</b>: ───iSwap───
      │
* <b>`1`</b>: ───iSwap───
```

```
>>> print(cirq.quirk_url_to_circuit(
...     'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
...     extra_cell_makers=[
...         cirq.interop.quirk.cells.CellMaker(
...             identifier='iswap',
...             size=2,
...             maker=lambda args: cirq.ISWAP(*args.qubits))
...     ]))
* <b>`0`</b>: ───iSwap───
      │
* <b>`1`</b>: ───iSwap───
```


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The parsed circuit.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
Invalid circuit URL, or circuit would be larger than
`max_operations_count`.
</td>
</tr>
</table>

