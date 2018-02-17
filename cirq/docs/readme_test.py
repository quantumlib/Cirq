import os
import re


def test_readme_code_snippets_execute():
    # Get the contents of the README.md file at the project root.
    readme_path = os.path.join(
        os.path.split(__file__)[0],  # Start at this file's directory.
        '..', '..', 'cirq', 'docs',  # Hacky check that we're under cirq/docs/.
        '..', '..', 'README.md')     # Get the readme two levels up.
    try:
        with open(readme_path, 'r') as f:
            readme_content = f.read()
    except IOError:
        # Readme not found. Not great.. but no need to test that it runs!
        return

    # Find snippets of code, and execute them. They should finish.
    for snippet in re.findall("\n```python(.*?)\n```\n",
                              readme_content,
                              re.MULTILINE | re.DOTALL):
        exec(snippet, {})
