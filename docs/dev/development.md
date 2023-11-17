# Contribute

This document is a summary of how to do various tasks one runs into as a developer of Cirq.
Note that all commands assume a Debian environment, and all commands (except the initial repository cloning command) assume your current working directory is the cirq repo root.


## Cloning the repository

The simplest way to get a local copy of cirq that you can edit is by cloning Cirq's github repository:

```bash
git clone git@github.com:quantumlib/cirq.git
cd Cirq
```

## Recommended git setup

The following command will setup large refactoring revisions to be ignored, when using git blame.

```
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

Note that if you are using PyCharm, you might have to Restart & Invalidate Caches to have the change being picked up. 

## Docker
 You can build the stable and pre_release docker images with our `Dockerfile`.

```bash
    docker build -t cirq --target cirq_stable .
    docker run -it cirq python -c "import cirq_google; print(cirq_google.Sycamore23)"
```

```bash
    docker build -t cirq_pre --target cirq_pre_release .
    docker run -it cirq_pre python -c "import cirq_google; print(cirq_google.Sycamore23)"
```

If you want to contribute changes to Cirq, you will instead want to fork the repository and submit pull requests from your fork.



## Forking the repository

1. Fork the Cirq repo (Fork button in upper right corner of
[repo page](https://github.com/quantumlib/Cirq)).
Forking creates a new github repo at the location
https://github.com/USERNAME/cirq where `USERNAME` is
your github id.
1. Clone the fork you created to your local machine at the directory
where you would like to store your local copy of the code, and `cd` into the newly created directory.
    
   ```bash
    git clone git@github.com:USERNAME/cirq.git
    cd Cirq
   ```
   (Alternatively, you can clone the repository using the URL provided on your repo page under the green "Clone or Download" button)
1. Add a remote called ```upstream``` to git.
This remote will represent the main git repo for cirq (as opposed to the clone, which you just created, which will be the ```origin``` remote). 
This remote can be used to merge changes from Cirq's main repository into your local development copy.
   
    ```shell
    git remote add upstream https://github.com/quantumlib/cirq.git
    ```
   
    To verify the remote, run ```git remote -v```. You should see both the ```origin``` and ```upstream``` remotes.
1. Sync up your local git with the ```upstream``` remote:
   
    ```shell
    git fetch upstream
    ```
   
    You can check the branches that are on the ```upstream``` remote by
    running ```git remote -va``` or ```git branch -r```.
Most importantly you should see ```upstream/master``` listed.
1. Merge the upstream master into your local master so that it is up to date.
    
   ```shell
    git checkout master
    git merge upstream/master
   ```
    
At this point your local git master should be synced with the master from the main cirq repo.


## Setting up an environment

These instructions are primarily for linux-based environments that use the apt
package manager. 

0. First clone the repository, if you have not already done so.
See the previous section for instructions.


1. Install system dependencies.

    Make sure you have python 3.9 or greater.
    You can install most other dependencies via `apt-get`:

    ```bash
    cat apt-system-requirements.txt dev_tools/conf/apt-list-dev-tools.txt | xargs sudo apt-get install --yes
    ```
    
    This installs docker and docker-compose among other things. You may need to restart
    docker or configure permissions, see 
    [docker install instructions](https://docs.docker.com/engine/install/ubuntu/).
    Note that docker is necessary only for cirq_rigetti.

    There are some extra steps if protocol buffers are changed; see the next section.

2. Prepare a virtual environment including the dev tools (such as mypy).

    One of the system dependencies we installed was `virtualenvwrapper`, which makes it easy to create virtual environments.
    If you did not have `virtualenvwrapper` previously, you may need to re-open your terminal or run `source ~/.bashrc` before these commands will work:

    ```bash
    mkvirtualenv cirq-py3 --python=/usr/bin/python3
    workon cirq-py3
    python -m pip install --upgrade pip    
    python -m pip install -r dev_tools/requirements/dev.env.txt
    ```

    (When you later open another terminal, you can activate the virtualenv with `workon cirq-py3`.)

3. Check that the tests pass.

    ```bash
    ./check/pytest .
    ```

4. (**OPTIONAL**) include your development copy of cirq and its subpackages in your python path.

    ```bash
    source dev_tools/pypath
    ```
    
    or add it to the python path, but only in the virtualenv by first listing the modules
    
    ```bash
    python dev_tools/modules.py list 
    ```
    and then adding these to the virtualenv:
    ```bash
    add2virtualenv <paste modules from last command>
    ```
    (Typically `add2virtualenv` is not executable using xargs, so this two step process is necessary.)

## Editable installs 

If you want to pip install cirq in an editable fashion, you'll have to install it per module, e.g.: 

```
pip install -e ./cirq-core -e ./cirq-google -e ./cirq-ionq -e ./cirq-aqt
```

Note that `pip install -e .` will install the `cirq` metapackage only, and your code changes won't 
get picked up! 

## Protocol buffers

[Protocol buffers](https://developers.google.com/protocol-buffers) are used in Cirq for converting circuits, gates, and other objects into a standard form that can be written and read by other programs.
Cirq's protobufs live at [cirq-google/api/v2](https://github.com/quantumlib/Cirq/tree/master/cirq-google/cirq_google/api/v2) and may need to be changed or extended from time to time.

If any protos are updated, their dependents can be rebuilt by calling the script [dev_tools/build-protos.sh](https://github.com/quantumlib/Cirq/tree/master/dev_tools).
This script uses grpcio-tools and protobuf version 3.8.0 to generate the python proto api.

## Continuous integration and local testing

There are a few options for running continuous integration checks, varying from easy and fast to slow and reliable.

The simplest way to run checks is to invoke `pytest`, `pylint`, or `mypy` for yourself as follows:

```bash
pytest
pylint --rcfile=dev_tools/conf/.pylintrc cirq
mypy --config-file=dev_tools/conf/mypy.ini .
```

This can be a bit tedious, because you have to specify the configuration files each time.
A more convenient way to run checks is to via the scripts in the [check/](https://github.com/quantumlib/Cirq/tree/master/check) directory, which specify configuration arguments for you and cover more use cases:

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
against for an incremental check (e.g. in order to determine which files changed.)
If not specified, it defaults to the `upstream/master` branch if it exists, or
else the `origin/master` branch if it exists, or else the `master` branch.
The actual commit used for comparison is the `git merge-base` of the base
revision and the working directory.

The above scripts may not exactly match the results computed by the continuous integration builds run on Travis.
For example, you may be running an older version of `pylint` or `numpy`.
If you need to test against the actual continuous integration check, open up a pull request.
For this pull request you may want to mark it as `[Testing]` so that it is not reviewed.

### Writing docstrings and generating documentation

Cirq uses [Google style doc strings](http://google.github.io/styleguide/pyguide.html#381-docstrings) with a markdown flavor and support for latex.
Here is an example docstring:

```
def some_method(a: int, b: str) -> float:
    r"""One line summary of method.

    Additional information about the method, perhaps with some sort of latex
    equation to make it clearer:

        $$
        M = \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}
        $$

    Notice that this docstring is an r-string, since the latex has backslashes.
    We can also include example code:

        print(cirq_google.Sycamore) 

    You can also do inline latex like $y = x^2$ and inline code like
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

Cirq follows a modular design. Each module should specify their dependencies within their folder. See for example cirq-core/requirements.txt and cirq-google/requirements.txt.
In general we should try to keep dependencies as minimal as possible and if we have to add them, keep them as relaxed as possible instead of pinning to exact versions. If exact versions or constraints are known, those should be documented in form of a comment. 

### Development dependencies 

For local development: 

For a development environment there is a single file that installs all the module dependencies and all of the dev tools as well: dev_tools/requirements/dev.env.txt.
If this is too heavy weight for you, you can instead use dev_tools/requirements/deps/dev-tools.txt and the given module dependencies. 

For continuous integration: 

Each job might need different set of requirements and it would be inefficient to install a full blown dev env for every tiny job (e.g. mypy check). 
Instead in dev_tools/requirements create a separate <job>.env.txt and include the necessary tools in there. Requirements files can include each other, which is heavily leveraged in our requirements files in order to remove duplication.   

You can call the following utility to unroll the content of a file: 

```
python dev_tools/requirements/reqs.py dev_tools/requirements/dev.env.txt 
```

## Producing a pypi package

1. Do a dry run with test pypi.

    If you're making a release, you should have access to a test pypi account
    capable of uploading packages to cirq. Put its credentials into the environment
    variables `TEST_TWINE_USERNAME` and `TEST_TWINE_PASSWORD` then run

    ```bash
    ./dev_tools/packaging/publish-dev-package.sh EXPECTED_VERSION --test
    ```

    You must specify the EXPECTED_VERSION argument to match the version in [cirq/_version.py](https://github.com/quantumlib/Cirq/blob/master/cirq-core/cirq/_version.py), and it must contain the string `dev`.
    This is to prevent accidentally uploading the wrong version.

    The script will append the current date and time to the expected version number before uploading to test pypi.
    It will print out the full version that it uploaded.
    Take not of this value.

    Once the package has uploaded, verify that it works

    ```bash
    ./dev_tools/packaging/verify-published-package.sh FULL_VERSION_REPORTED_BY_PUBLISH_SCRIPT --test
   ```

    The script will create fresh virtual environments, install cirq and its dependencies, check that code importing cirq executes, and run the tests over the installed code. If everything goes smoothly, the script will finish by printing `VERIFIED`.

2. Do a dry run with prod pypi

    This step is essentially identical to the test dry run, but with production pypi.
    You should have access to a production pypi account capable of uploading packages to cirq.
    Put its credentials into the environment variables `PROD_TWINE_USERNAME` and `PROD_TWINE_PASSWORD` then run

    ```bash
    ./dev_tools/packaging/publish-dev-package.sh EXPECTED_VERSION --prod
    ```

    Once the package has uploaded, verify that it works

    ```bash
    ./dev_tools/packaging/verify-published-package.sh FULL_VERSION_REPORTED_BY_PUBLISH_SCRIPT --prod
   ```

    If everything goes smoothly, the script will finish by printing `VERIFIED`.

3. Set the version number in [cirq/_version.py](https://github.com/quantumlib/Cirq/blob/master/cirq-core/cirq/_version.py).

    Development versions end with `.dev` or `.dev#`.
    For example, `0.0.4.dev500` is a development version of the release version `0.0.4`.
    For a release, create a pull request turning `#.#.#.dev*` into `#.#.#` and a follow up pull request turning `#.#.#` into `(#+1).#.#.dev`.

4. Run [dev_tools/packaging/produce-package.sh](https://github.com/quantumlib/Cirq/blob/master/dev_tools/packaging/produce-package.sh) to produce pypi artifacts.

    ```bash
    ./dev_tools/packaging/produce-package.sh dist
    ```

    The output files will be placed in the directory `dist/`.

5. Create a github release.

    Describe major changes (especially breaking changes) in the summary.
    Make sure you point the tag being created at the one and only revision with the non-dev version number.
    Attach the package files you produced to the release.

6. Upload to pypi.

    You can use a tool such as `twine` for this.
    For example:

    ```bash
    twine upload -u "${PROD_TWINE_USERNAME}" -p "${PROD_TWINE_PASSWORD}" dist/*
    ```

    You should then run the verification script to check that the uploaded package works:

    ```bash
    ./dev_tools/packaging/verify-published-package.sh VERSION_YOU_UPLOADED --prod
   ```

    And try it out for yourself:

    ```bash
    python -m pip install cirq
    python -c "import cirq; print(cirq_google.Sycamore)"
    python -c "import cirq; print(cirq.__version__)"
    ```
