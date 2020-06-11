<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.read_json" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.read_json

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/json_serialization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Read a JSON file that optionally contains cirq objects.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.protocols.json_serialization.read_json`, `cirq.read_json`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.read_json(
    file_or_fn: Union[None, IO, pathlib.Path, str] = None,
    *,
    json_text: Optional[str] = None,
    resolvers: Optional[List[Callable[[str], Union[None, Type]]]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`file_or_fn`
</td>
<td>
A filename (if a string or `pathlib.Path`) to read from, or
an IO object (such as a file or buffer) to read from, or `None` to
indicate that `json_text` argument should be used. Defaults to
`None`.
</td>
</tr><tr>
<td>
`json_text`
</td>
<td>
A string representation of the JSON to parse the object from,
or else `None` indicating `file_or_fn` should be used. Defaults to
`None`.
</td>
</tr><tr>
<td>
`resolvers`
</td>
<td>
A list of functions that are called in order to turn
the serialized `cirq_type` string into a constructable class.
By default, top-level cirq objects that implement the SupportsJSON
protocol are supported. You can extend the list of supported types
by pre-pending custom resolvers. Each resolver should return `None`
to indicate that it cannot resolve the given cirq_type and that
the next resolver should be tried.
</td>
</tr>
</table>

