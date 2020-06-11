<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.engine.engine.EngineContext" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="copy"/>
</div>

# cirq.google.engine.engine.EngineContext

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Context for running against the Quantum Engine API. Most users should

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.engine.engine.EngineContext(
    proto_version: Optional[<a href="../../../../cirq/google/ProtoVersion.md"><code>cirq.google.ProtoVersion</code></a>] = None,
    service_args: Optional[Dict] = None,
    verbose: Optional[bool] = None,
    client: "Optional[engine_client.EngineClient]" = None,
    timeout: Optional[int] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
simply create an Engine object instead of working with one of these
directly.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`proto_version`
</td>
<td>
The version of cirq protos to use. If None, then
ProtoVersion.V2 will be used.
</td>
</tr><tr>
<td>
`service_args`
</td>
<td>
A dictionary of arguments that can be used to
configure options on the underlying client.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
Suppresses stderr messages when set to False. Default is
true.
</td>
</tr><tr>
<td>
`timeout`
</td>
<td>
Timeout for polling for results, in seconds.  Default is
to never timeout.
</td>
</tr>
</table>



## Methods

<h3 id="copy"><code>copy</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/engine.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy() -> "EngineContext"
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: _SupportsValueEquality
) -> bool
</code></pre>




<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/value_equality.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other: _SupportsValueEquality
) -> bool
</code></pre>






