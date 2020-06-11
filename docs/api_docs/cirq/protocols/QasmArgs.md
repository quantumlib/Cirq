<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.QasmArgs" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="check_unused_args"/>
<meta itemprop="property" content="convert_field"/>
<meta itemprop="property" content="format"/>
<meta itemprop="property" content="format_field"/>
<meta itemprop="property" content="get_field"/>
<meta itemprop="property" content="get_value"/>
<meta itemprop="property" content="parse"/>
<meta itemprop="property" content="validate_version"/>
<meta itemprop="property" content="vformat"/>
</div>

# cirq.protocols.QasmArgs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/qasm.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>





<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.QasmArgs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.QasmArgs(
    precision: int = 10,
    version: str = '2.0',
    qubit_id_map: Dict['cirq.Qid', str] = None,
    meas_key_id_map: Dict[str, str] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="check_unused_args"><code>check_unused_args</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>check_unused_args(
    used_args, args, kwargs
)
</code></pre>




<h3 id="convert_field"><code>convert_field</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert_field(
    value, conversion
)
</code></pre>




<h3 id="format"><code>format</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>format(
    *args, **kwargs
)
</code></pre>




<h3 id="format_field"><code>format_field</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/qasm.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>format_field(
    value: Any,
    spec: str
) -> str
</code></pre>

Method of string.Formatter that specifies the output of format().


<h3 id="get_field"><code>get_field</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_field(
    field_name, args, kwargs
)
</code></pre>




<h3 id="get_value"><code>get_value</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_value(
    key, args, kwargs
)
</code></pre>




<h3 id="parse"><code>parse</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>parse(
    format_string
)
</code></pre>




<h3 id="validate_version"><code>validate_version</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/qasm.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>validate_version(
    *supported_versions
) -> None
</code></pre>




<h3 id="vformat"><code>vformat</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vformat(
    format_string, args, kwargs
)
</code></pre>






