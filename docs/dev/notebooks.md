# Notebooks guidelines 

Our guides and tutorials are frequently written using iPython Notebooks. The notebooks require specific formatting, are continuosly tested (when possible) and we have a specific process to manage the lifecycle of a notebook before and after a Cirq release.    

## Formatting 

Formatting is easy, the script `check/nbformat` should tell you if your notebooks are formatted or not.
You can apply the changes in one go with `check/nbformat --apply`. It is recommended to add this to you [git pre-commit hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks), to save feedback time and CI resources. 

## Header

We also expect a standard header to be included in all of our notebooks: 
- the links to colab, github and the main site (quantumai.google/cirq)
- optional package installation (you can assume Colab dependencies exist)
 

## Testing 

Those notebooks that don't have any external dependencies (e.g. authentication) can be executed in an isolated environment are being tested on a continuous basis. 
See the `dev_tools/notebooks` directory for the two tests: 
- notebook_tests.py - to test notebooks against the current branch
- isolated_notebook_tests.py - to test notebooks against the latest released version of Cirq

## Lifecycle 

Notebooks are handled differently based on whether they rely on features in the pre-release build of cirq or not. 

Pre-release notebooks: 
 - mark the notebook at the top that `Note: this notebook relies on unreleased Cirq features. If you want to try these feature, make sure you install cirq via pip install cirq --pre`. 
 - use `pip install cirq —pre`  in the installation instructions 
 - make sure dev_tools/notebooks/test_notebooks.py covers the notebook 
 - exclude the notebook from the dev_tools/notebooks/isolated_notebook_test.py by adding it to `NOTEBOOKS_DEPENDING_ON_UNRELEASED_FEATURES`

After the Cirq release - for all unreleased notebooks we change all the above accordingly in bulk for all the notebooks: 
 - remove the pre-release notices
 - change `pip install cirq —pre` to `pip install cirq`
 - remove the exclusions in dev_tools/notebooks/isolated_notebook_test.py by making `NOTEBOOKS_DEPENDING_ON_UNRELEASED_FEATURES=[]`
 
As all the notebooks have been tested continuously up to this point, the post release notebook PR should pass without issues. 
 
If a released notebook needs to be modified to cater for unreleased functionality, then it will again become a pre-release notebook. 