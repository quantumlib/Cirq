<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.engine_from_environment" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.engine_from_environment

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/engine/env_config.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns an Engine instance configured using environment variables.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.engine.engine_from_environment`, `cirq.google.engine.env_config.engine_from_environment`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.engine_from_environment() -> <a href="../../cirq/google/Engine.md"><code>cirq.google.Engine</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

If the environment variables are set, but incorrect, an authentication
failure will occur when attempting to run jobs on the engine.

Required Environment Variables:
    QUANTUM_ENGINE_PROJECT: The name of a google cloud project, with the
        quantum engine enabled, that you have access to.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`EnvironmentError`
</td>
<td>
The environment variables are not set.
</td>
</tr>
</table>

