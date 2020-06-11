<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.VirtualTag" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
</div>

# cirq.ops.VirtualTag

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/tags.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A TaggedOperation tag indicating that the operation is virtual.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.VirtualTag`, `cirq.ops.tags.VirtualTag`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

Operations marked with this tag are presumed to have zero duration of their
own, although they may have a non-zero duration if run in the same Moment
as a non-virtual operation.

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/ops/tags.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




