# Contribute

This document is a summary of how to do various tasks that one might need to do as a developer of Cirq.
Note that all commands assume a Debian environment, and all commands (except the initial repository cloning command) assume your current working directory is your local Cirq repository root directory.


## Cloning the repository

The simplest way to get a local copy of Cirq that you can edit is by cloning Cirq's GitHub repository:

```bash
git clone https://github.com/quantumlib/Cirq.git
cd Cirq
```

## Recommended git setup

The following command will set up large refactoring revisions to be ignored, when using git blame.

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

Note that if you are using PyCharm, you might have to use the command Restart & Invalidate Caches to have the change be picked up.

## Forking the repository

1. Fork the Cirq repo (Fork button in upper right corner of
[repo page](https://github.com/quantumlib/Cirq)).
Forking creates a new GitHub repo at the location
https://github.com/USERNAME/cirq where `USERNAME` is
your GitHub id.
1. Clone the fork you created to your local machine at the directory
where you would like to store your local copy of the code, and `cd` into the newly created directory.

   ```bash
    git clone https://github.com/USERNAME/Cirq.git
    cd Cirq
   ```
   (Alternatively, you can clone the repository using the URL provided on your repo page under the green "Clone or Download" button)
1. Add a remote called ```upstream``` to git.
This remote will represent the main git repo for Cirq (as opposed to the clone, which you just created, which will be the ```origin``` remote).
This remote can be used to merge changes from Cirq's main repository into your local development copy.

    ```shell
    git remote add upstream https://github.com/quantumlib/Cirq.git
    ```

    To verify the remote, run ```git remote -v```. You should see both the ```origin``` and ```upstream``` remotes.
1. Sync up your local git with the ```upstream``` remote:

    ```shell
    git fetch upstream
    ```

    You can check the branches that are on the ```upstream``` remote by
    running `git ls-remote --branches upstream` or `git branch -r`.
Most importantly you should see ```upstream/main``` listed.
1. Merge the upstream main into your local main so that it is up to date.

   ```shell
    git checkout main
    git merge upstream/main
   ```

At this point your local git main should be synced with the main from the main Cirq repo.


## Setting up an environment

These instructions are primarily for Linux-based environments that use the `apt`
package manager.

0. First clone the repository, if you have not already done so.
   See the previous section for instructions.

1. Install system dependencies.

    Make sure you have Python 3.11 or greater.
    You can install most other dependencies via `apt-get`:

    ```bash
    cat apt-system-requirements.txt dev_tools/conf/apt-list-dev-tools.txt | xargs sudo apt-get install --yes
    ```

    There are some extra steps if Protocol Buffers are changed; see the next section.


2.  One of the system dependencies we installed was `virtualenvwrapper`, which makes it easy to create virtual environments. If you did not have `virtualenvwrapper` previously then to complete the setup of virtualenvwrapper,  the following lines must be added to your ~/.bashrc or ~/.zshrc file. Once the file is saved, you will need to either open a new terminal session or execute source ~/.bashrc to activate the new configuration.

    ```bash
    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
    ```

3. Prepare a Python virtual environment that includes the Cirq dev tools (such as Mypy).

    ```bash
    mkvirtualenv cirq-py3 --python=/usr/bin/python3
    workon cirq-py3
    python -m pip install --upgrade pip
    python -m pip install -r dev_tools/requirements/dev.env.txt
    ```

    (When you later open another terminal, you can activate the virtualenv with `workon cirq-py3`.)

    **Note**: Some highly managed or customized devices have configurations that interfere with `virtualenv`.
    In that case, [anaconda](https://www.anaconda.com/) environments may be a better choice.

4. Check that the tests pass.

    ```bash
    ./check/pytest .
    ```

5. (**OPTIONAL**) include your development copy of Cirq and its subpackages in your Python path.

    ```bash
    source dev_tools/pypath
    ```

    or add it to the Python path, but only in the virtualenv by first listing the modules

    ```bash
    python dev_tools/modules.py list
    ```
    and then adding these to the virtualenv:
    ```bash
    add2virtualenv <paste modules from last command>
    ```
    (Typically `add2virtualenv` is not executable using `xargs`, so this two step process is necessary.)

## Editable installs

If you want to pip install Cirq in an editable fashion, you'll have to install it per module, e.g.:

```bash
pip install -e ./cirq-core -e ./cirq-google -e ./cirq-ionq -e ./cirq-aqt
```

Note that `pip install -e .` will install the `cirq` metapackage only, and your code changes won't
get picked up!

## Protocol buffers

[Protocol buffers](https://developers.google.com/protocol-buffers) ("protobufs") are used in Cirq for converting circuits, gates, and other objects into a standard form that can be written and read by other programs.
Cirq's protobufs live at [cirq-google/api/v2](https://github.com/quantumlib/Cirq/tree/main/cirq-google/cirq_google/api/v2) and may need to be changed or extended from time to time.

If any protos are updated, their dependents can be rebuilt by calling the script
[dev_tools/build-protos.sh](https://github.com/quantumlib/Cirq/blob/main/dev_tools/build-protos.sh).
This script uses the `grpcio-tools` package to generate the Python proto API.

## Continuous integration and local testing

There are a few options for running continuous integration checks, varying from easy and fast to slow and reliable.

The simplest way to run checks is to invoke `pytest`, `pylint`, or `mypy` for yourself as follows:

```bash
pytest
pylint --rcfile=dev_tools/conf/.pylintrc cirq
mypy --config-file=dev_tools/conf/mypy.ini .
```

This can be a bit tedious, because you have to specify the configuration files each time.
A more convenient way to run checks is to via the scripts in the [check/](https://github.com/quantumlib/Cirq/tree/main/check) directory, which specify configuration arguments for you and cover more use cases:

- **Fast checks (complete in seconds or tens of seconds)**

    - Check or apply code formatting to changed lines:

         ```bash
         ./check/format-incremental [--apply] [BASE_REVISION]
         ```

    - Run tests associated with changed files:

        ```bash
        ./check/pytest-changed-files [BASE_REVISION]
        ```

    - Run tests embedded in docstrings:

        ```bash
        ./check/doctest
        ```

    - Compute incremental coverage using only tests associated with changed files:

        ```bash
        ./check/pytest-changed-files-and-incremental-coverage [BASE_REVISION]
        ```

        Note: this check is stricter than the incremental coverage check we
        actually enforce, where lines may be covered by tests in
        unassociated files.

    - Type checking:

        ```bash
        ./check/mypy [files-and-flags-for-mypy]
        ```

    - Miscellaneous checks:

        ```bash
        ./check/misc
        ```

        (Currently just checks that nothing outside `cirq.contrib` references
        anything inside `cirq.contrib`.)

- **Slow checks (each takes a few minutes)**

    - Run all tests:

        ```bash
        ./check/pytest [files-and-flags-for-pytest]
        ```

    - Check for lint:

        ```bash
        ./check/pylint [files-and-flags-for-pylint]
        ```

    - Compute incremental coverage:

        ```bash
        ./check/pytest-and-incremental-coverage [BASE_REVISION]
        ```

    - Run all continuous integration checks:

        ```bash
        ./check/all [BASE_REVISION] [--only-changed-files] [--apply-format-changes]
        ```

        If `--only-changed-files` is set, checks that can will focus down to
        just files that were changed (trading accuracy for speed).

In the above, `[BASE_REVISION]` controls what commit is being compared
against for an incremental check (e.g., in order to determine which files changed).
If not specified, it defaults to the `upstream/main` branch if it exists, or
else the `origin/main` branch if it exists, or else the `main` branch.
The actual commit used for comparison is the `git merge-base` of the base
revision and the working directory.

The above scripts may not exactly match the results computed by the continuous integration workflows on GitHub.
For example, you may be running an older version of `pylint` or `numpy`.
If you need to test against the actual continuous integration check, open up a pull request.
For this pull request you may want to open it in a draft mode or
mark it as `[Testing]` so that it is not reviewed.

### Writing docstrings and generating documentation

Cirq uses [Google style doc strings](http://google.github.io/styleguide/pyguide.html#381-docstrings) with a Markdown flavor and support for LaTeX.
Here is an example docstring:

```python
def some_method(a: int, b: str) -> float:
    r"""One line summary of method.

    Additional information about the method, perhaps with some sort of LaTeX
    equation to make it clearer:

        $$
        M = \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}
        $$

    Notice that this docstring is an r-string, since the LaTeX has backslashes.
    We can also include example code:

        print(cirq_google.Sycamore)

    You can also do inline LaTeX like $y = x^2$ and inline code like
    `cirq.unitary(cirq.X)`.

    And of course there's the standard sections.

    Args:
        a: The first argument.
        b: Another argument.

    Returns:
        An important value.

    Raises:
        ValueError: The value of `a` wasn't quite right.
    """
```

## Dependencies

### Production dependencies

Cirq follows a modular design. Each module should specify their dependencies in files within their folder. See, for example, the files `cirq-core/requirements.txt` and `cirq-google/requirements.txt`.
In general, we should try to keep dependencies as minimal as possible and if we have to add them, keep them as relaxed as possible instead of pinning to exact versions. If exact versions or constraints are known, those should be documented in form of a comment.

### Development dependencies

For local development:

For a development environment there is a single file that installs all the module dependencies and all of the dev tools as well: `dev_tools/requirements/dev.env.txt`.
If this is too heavy weight for you, you can instead use `dev_tools/requirements/deps/dev-tools.txt` and the given module dependencies.

For continuous integration:

Each job might need different set of requirements and it would be inefficient to install a full-blown dev env for every tiny job (e.g. `mypy` check).
Instead, in the directory `dev_tools/requirements`, create a separate `<job>.env.txt` and include the necessary tools in there. Requirements files can include each other, which is heavily leveraged in our requirements files in order to remove duplication.

You can call the following utility to unroll the content of a file:

```bash
python dev_tools/requirements/reqs.py dev_tools/requirements/dev.env.txt
```
