## Development

This document is a summary of how to do various tasks one runs into as a developer of Cirq.
Note that all commands assume a Debian environment, and all commands (except the initial repository cloning command) assume your current working directory is the cirq repo root.


### Cloning the repository

The simplest way to get a local copy of cirq that you can edit is by cloning Cirq's github repository:

```bash
git clone git@github.com:quantumlib/cirq.git
cd cirq
```

If you want to contribute changes to Cirq, you will instead want to fork the repository and submit pull requests from your fork.



### Forking the repository

1. Fork the Cirq repo (Fork button in upper right corner of
[repo page](https://github.com/quantumlib/Cirq)).
Forking creates a new github repo at the location
```https://github.com/USERNAME/cirq``` where ```USERNAME``` is
your github id.
1. Clone the fork you created to your local machine at the directory
where you would like to store your local copy of the code, and `cd` into the newly created directory.
    ```bash
    git clone git@github.com:USERNAME/cirq.git
    cd cirq
    ```
    (Alternatively, you can clone the repository using the URL provided
    on your repo page under the green "Clone or Download" button)
1. Add a remote called ```upstream``` to git.
This remote will represent the main git repo for cirq (as opposed to the clone, which you just created, which will be the ```origin``` remote). 
This remote can be used to merge changes from Cirq's main repository into your local development copy.
    ```shell
    git remote add upstream https://github.com/quantumlib/cirq.git
    ```
    To verify the remote, run ```git remote -v```.
    You should see both the ```origin``` and ```upstream``` remotes.
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


### Setting up an environment

0. First clone the repository, if you have not already done so.
See the previous section for instructions.


1. Install system dependencies.

    Make sure you have python 3.5 or greater.
    You can install most other dependencies via `apt-get`:

    ```bash
    cat apt-system-requirements.txt dev_tools/conf/apt-list-dev-tools.txt | xargs sudo apt-get install --yes
    ```

    If you change protocol buffers you will need to regenerate the proto files, so you should
    install the protocol buffer compiler. Instructions for this can be found
    [here](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md).

2. Prepare a virtual environment including the dev tools (such as mypy).

    One of the system dependencies we installed was `virtualenvwrapper`, which makes it easy to create virtual environments.
    If you did not have `virtualenvwrapper` previously, you may need to re-open your terminal or run `source ~/.bashrc` before these commands will work:

    ```bash
    mkvirtualenv cirq-py3 --python=/usr/bin/python3
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r dev_tools/conf/pip-list-dev-tools.txt
    ```

    (When you later open another terminal, you can activate the virtualenv with `workon cirq-py3`.)

3. Check that the tests pass.

    ```bash
    pytest .
    ```

4. (**OPTIONAL**) include your development copy of cirq in your python path.

    ```bash
    PYTHONPATH="$(pwd)":"${PYTHONPATH}"
    ```
    
    or add it to the python path, but only in the virtualenv.
    
    ```bash
    add2virtualenv ./
    ```


### Running continuous integration checks locally

There are a few options for running continuous integration checks, varying from easy and fast to slow and reliable.

The simplest way to run checks is to invoke `pytest`, `pylint`, or `mypy` for yourself as follows:

```bash
pytest
pylint --rcfile=dev_tools/conf/.pylintrc cirq
mypy --config-file=dev_tools/conf/mypy.ini .
```

This can be a bit tedious, because you have to specify the configuration files each time.
A more convenient way to run checks is to via the scripts in the [check/](/check) directory, which specify configuration arguments for you and cover more use cases:

```bash
# Run all tests in the repository.
./check/pytest [files-and-flags-for-pytest]

# Check all relevant files in the repository for lint.
./check/pylint [files-and-flags-for-pylint]

# Typecheck all python files in the repository.
./check/mypy [files-and-flags-for-mypy]

# Transpile to python 2 and run tests.
./check/pytest2  # Note: you must be in a python 2 virtual env to run this.

# Compute incremental coverage vs master (or a custom revision of your choice).
./check/pytest-and-incremental-coverage [BASE_REVISION]

# Only run tests associated with files that have changed when diffed vs master (or a custom revision of your choice).
./check/pytest-changed-files [BASE_REVISION]
```

The above scripts are convenient and reasonably fast, but they often won't exactly match the results computed by the continuous integration builds run on travis.
For example, you may be running an older version of `pylint` or `numpy`.
In order to run a check that is significantly more likely to agree with the travis builds, you can use the [continuous-integration/check.sh](/continuous-integration/check.sh) script:

```bash
./continuous-integration/check.sh
```

This script will create (temporary) virtual environments, do a fresh install of all relevant dependencies, transpile the python 2 code, and run all relevant checks within those clean environments.
Note that creating the virtual environments takes time, and prevents some caching mechanisms from working, so `continuous-integration/check.sh` is significantly slower than the simpler check scripts.
When using this script, you can run a subset of the checks using the ```--only``` flag.
This flag value can be `pylint`, `typecheck`, `pytest`, `pytest2`, or `incremental-coverage`.


### Producing the Python 2.7 code

Run [dev_tools/python2.7-generate.sh](/dev_tools/python2.7-generate.sh) to transpile cirq's python 3 code into python 2.7 code:

```bash
./dev_tools/python2.7-generate.sh [output_dir] [input_dir] [virtual_env_with_3to2]
```

If you don't specify any arguments then the input directory will be the current
working directory, the output directory will be `python2.7-output` within the
current directory, and `3to2` will be invoked in the current environment.

The script fails with no effects if the output directory already exists.


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

        print(cirq.google.Foxtail)

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

Documentation is generated automatically by readthedocs when pushing to `master`, but you can also generated a local copy by running:

```bash
dev_tools/build-docs.sh
```

The HTML output will go into the `docs/_build` directory.


### Producing a pypi package

1. Do a dry run with test pypi.

    If you're making a release, you should have access to a test pypi account
    capable of uploading packages to cirq. Put its credentials into the environment
    variables `TEST_TWINE_USERNAME` and `TEST_TWINE_PASSWORD` then run

    ```bash
    ./dev_tools/packaging/publish-dev-package.sh EXPECTED_VERSION --test
    ```

    You must specify the EXPECTED_VERSION argument to match the version in `cirq/_version.py`, and it must contain the string `dev`.
    This is to prevent accidentally uploading the wrong version.

    The script will append the current date and time to the expected version number before uploading to test pypi.
    It will print out the full version that it uploaded.
    Take not of this value.

    Once the package has uploaded, verify that it works

    ```bash
    ./dev_tools/packaging/verify-published-package.sh FULL_VERSION_REPORTED_BY_PUBLISH_SCRIPT --test
   ```

    The script will create fresh virtual environments, install cirq and its dependencies, check that code importing cirq executes, and run the tests over the installed code.
    It will do this for both python 2 and python 3.
    If everything goes smoothly, the script will finish by printing `VERIFIED`.

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

3. Set the version number in [cirq/_version.py](/cirq/_version.py).

    Development versions end with `.dev` or `.dev#`.
    For example, `0.0.4.dev500` is a development version of the release version `0.0.4`.
    For a release, create a pull request turning `#.#.#.dev*` into `#.#.#` and a follow up pull request turning `#.#.#` into `(#+1).#.#.dev`.

4. Run [dev_tools/packaging/produce-package.sh](/dev_tools/packaging/produce-package.sh) to produce pypi artifacts.

    ```bash
    dev_tools/packaging/produce-package.sh dist
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
    pip install cirq
    python -c "import cirq; print(cirq.google.Foxtail)"
    python -c "import cirq; print(cirq.__version__)"
    ```
