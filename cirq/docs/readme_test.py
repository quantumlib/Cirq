import inspect
import os
import re


def test_readme_code_snippets_execute():
    # Get the contents of the README.md file at the project root.
    docs_directory = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    assert docs_directory.lower().endswith('/cirq/docs')
    project_directory = docs_directory[:-len('/cirq/docs')]
    readme_path = os.path.join(project_directory, 'README.md')
    with open(readme_path, 'r') as f:
        readme_content = f.read()

    # Find snippets of code, and execute them. They should finish.
    for snippet in re.findall("\n```python(.*?)\n```\n",
                              readme_content,
                              re.MULTILINE | re.DOTALL):
        exec(snippet, {})
