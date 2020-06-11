<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_eigengate_implements_consistent_protocols" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_eigengate_implements_consistent_protocols

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/consistent_protocols.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Checks that an EigenGate subclass is internally consistent and has a

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.consistent_protocols.assert_eigengate_implements_consistent_protocols`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_eigengate_implements_consistent_protocols(
    eigen_gate_type: Type[<a href="../../cirq/ops/EigenGate.md"><code>cirq.ops.EigenGate</code></a>],
    *,
    exponents: Sequence[<a href="../../cirq/value/TParamVal.md"><code>cirq.value.TParamVal</code></a>] = (0, 1, -1, 0.25, -0.5, 0.1, sympy.Symbol('s')),
    global_shifts: Sequence[float] = (0, -0.5, 0.1),
    qubit_count: Optional[int] = None,
    ignoring_global_phase: bool = False,
    setup_code: str = 'import cirq\nimport numpy as np\nimport sympy',
    global_vals: Optional[Dict[str, Any]] = None,
    local_vals: Optional[Dict[str, Any]] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->
good __repr__.