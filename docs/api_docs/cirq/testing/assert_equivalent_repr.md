<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_equivalent_repr" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_equivalent_repr

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/equivalent_repr_eval.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Checks that eval(repr(v)) == v.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.equivalent_repr_eval.assert_equivalent_repr`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_equivalent_repr(
    value: Any,
    *,
    setup_code: str = 'import cirq\nimport numpy as np\nimport sympy\nimport pandas as pd\n',
    global_vals: Optional[Dict[str, Any]] = None,
    local_vals: Optional[Dict[str, Any]] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
A value whose repr should be evaluatable python
code that produces an equivalent value.
</td>
</tr><tr>
<td>
`setup_code`
</td>
<td>
Code that must be executed before the repr can be evaluated.
Ideally this should just be a series of 'import' lines.
</td>
</tr><tr>
<td>
`global_vals`
</td>
<td>
Pre-defined values that should be in the global scope when
evaluating the repr.
</td>
</tr><tr>
<td>
`local_vals`
</td>
<td>
Pre-defined values that should be in the local scope when
evaluating the repr.
</td>
</tr>
</table>

