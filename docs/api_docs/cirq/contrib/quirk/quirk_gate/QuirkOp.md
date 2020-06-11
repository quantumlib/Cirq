<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.contrib.quirk.quirk_gate.QuirkOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="controlled"/>
</div>

# cirq.contrib.quirk.quirk_gate.QuirkOp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/quirk/quirk_gate.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An operation as understood by Quirk's parser.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.contrib.quirk.quirk_gate.QuirkOp(
    can_merge: bool = True,
    *keys
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Basically just a series of text identifiers for each qubit, and some rules
for how things can be combined.

## Methods

<h3 id="controlled"><code>controlled</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/contrib/quirk/quirk_gate.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>controlled(
    control_count: int = 1
) -> "QuirkOp"
</code></pre>






