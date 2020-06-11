<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_logs" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_logs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/logs.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A context manager for testing logging and warning events.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.logs.assert_logs`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_logs(
    *matches,
    count: int = 1,
    level: int = logging.WARNING,
    capture_warnings: bool = True
) -> ContextManager[List[logging.LogRecord]]
</code></pre>



<!-- Placeholder for "Used in" -->

To use this one wraps the code that is to be tested for log events within
the context of this method's return value:

    with assert_logs(count=2, 'first_match', 'second_match') as logs:
        <code that produces python logs>

This captures the logging that occurs in the context of the with statement,
asserts that the number of logs is equal to `count`, and then asserts that
all of the strings in `matches` appear in the messages of the logs.
Further, the captured logs are accessible as `logs` and further testing
can be done beyond these simple asserts.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`matches`
</td>
<td>
Each of these is checked to see if they match, as a substring,
any of the captures log meassages.
</td>
</tr><tr>
<td>
`count`
</td>
<td>
The expected number of messages in logs. Defaults to 1.
</td>
</tr><tr>
<td>
`level`
</td>
<td>
The level at which to capture the logs. See the python logging
module for valid levels. By default this captures at the
`logging.WARNING` level, so this does not capture `logging.INFO`
or `logging.DEBUG` logs by default.
</td>
</tr><tr>
<td>
`capture_warnings`
</td>
<td>
Whether warnings from the python's `warnings` module
are redirected to the logging system and captured.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A ContextManager that can be entered into which captures the logs
for code executed within the entered context. This ContextManager
checks that the asserts for the logs are true on exit.
</td>
</tr>

</table>

