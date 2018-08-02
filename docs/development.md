## Development

This document is a summary of how to do various tasks one runs into as a
developer of cirq.
Note that all commands assume a Debian environment, and all commands (except
the initial repository cloning command) assume you are at the cirq repo root.


### Cloning the repository

You can create a local version of this repository by running:

    ```bash
    git clone git@github.com:quantumlib/Cirq.git
    cd Cirq
    ```

This will allow you to use the Cirq library and build your own applications
using this framework.

If instead, you wish to do development on Cirq itself and contribute back to
the community, follow the directions below.

### Forking the repository

If do not plan to contribute back to the Cirq project and only wish to use
the Cirq framework to build your own quantum programs and circuits, you can
skip this section.

1. Fork the Cirq repo (Fork button in upper right corner of
[repo page](https://github.com/quantumlib/Cirq)).
Forking creates a new github repo at the location
```https://github.com/USERNAME/cirq``` where ```USERNAME``` is
your github id.
1. Clone the fork you created to your local machine at the directory
where you would like to store your local copy of the code.
    ```shell
    git clone git@github.com:USERNAME/cirq.git
    ```
    (Alternatively, you can clone the repository using the URL provided
    on your repo page under the green "Clone or Download" button)
1. Change directory to cirq.
   ```shell
   cd cirq
   ```
1. Add a remote called ```upstream``` to git.  This remote will represent
the main git repo for cirq (as opposed to the clone, which you just
created, which will be the ```origin``` remote).  This remote can be used
to sync your local git repos with the main git repo for cirq.
    ```shell
    git remote add upstream https://github.com/quantumlib/cirq.git
    ```
    To verify the remote run ```git remote -v``` and you should see both
    the ```origin``` and ```upstream``` remotes.
1. Sync up your local git with the ```upstream``` remote:
    ```shell
    git fetch upstream
    ```
    You can check the branches that are on the ```upstream``` remote by
    running ```git remote -va``` or ```git branch -r```.
Most importantly you should see ```upstream/master``` listed.
1. Merge the upstream master into your local master so that
it is up to date
    ```shell
    git checkout master
    git merge upstream/master
    ```
    At this point your local git master should be synced with the master
    from the main cirq repo.


### Setting up an environment.

0. First clone the repository, if you have not already done so.
See the previous section for instructions.


1. Install system dependencies.

    Make sure you have python 3.5 or greater.
    You can install most other dependencies via `apt-get`:

    ```bash
    cat apt-dev-requirements.txt apt-runtime-requirements.txt | xargs sudo apt-get install --yes
    ```

    Unfortunately, as of this writing, v3.5 of the [protobuf compiler](https://github.com/google/protobuf) is not installable via `apt-get`.
    You *can* skip this dependency.
    It is only needed when producing the transpiled python 2.7 code for local testing.
    The installation process is technical; requiring you to either build from source or manually installing pre-built binaries onto your system.
    If you want to be able to run the python 2.7 tests on your machine, see ["Protobuf Compiler Installation" on the google/protobuf github repository](https://github.com/google/protobuf#protocol-compiler-installation) for details.


2. Prepare a virtual environment with the dev requirements.

    One of the system dependencies we installed was `virtualenvwrapper`, which makes it easy to create virtual environment.
    If you did not have `virtualenvwrapper` previously, you may need to re-open your terminal or run `source ~/.bashrc` before these commands will work:

    ```bash
    mkvirtualenv cirq-py3 --python=/usr/bin/python3
    pip install --upgrade pip
    pip install -r dev-requirements.txt
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


### Running continuous integration checks locally

There are a few options for running continuous integration checks, varying from easy and fast to slow and reliable.

The simplest and fastest way to run checks is to invoke `pytest`, `pylint`, or `mypy` for yourself as follows:

```bash
pytest .
pylint --rcfile=continuous-integration/.pylintrc cirq  # note: only tests the cirq subdirectory
mypy --config-file=continuous-integration/mypy.ini .
```

But note that this will not check the incremental coverage or run the python 2 tests, and that during continuous integration pylint is run against more than just the cirq subdirectory.

The script [continuous-integration/simple_check.sh](/continuous-integration/simple_check.sh) will run pytest, mypy, pylint (on all the necessary files) and the incremental coverage check:

```bash
bash continuous-integration/simple_check.sh
```

But this script doesn't guarantee your test environment is up to date, and it does not transpile and test the python 2 code.

Use the script [continuous-integration/check.sh](/continuous-integration/check.sh) to do a proper full local check.
This script will create (temporary) virtual environments, do a fresh install of all relevant dependencies, transpile the python 2 code, and run all relevant checks.

```bash
bash continuous-integration/check.sh
```

You can run a subset of the checks using the ```--only``` flag.
This flag value can be `pylint`, `typecheck`, `pytest`, `pytest2`, or `incremental-coverage`.

### Producing the Python 2.7 code

Run [python2.7-generate.sh](/python2.7-generate.sh) to transpile cirq's python 3 code into python 2.7 code:

```bash
bash python2.7-generate.sh [output_dir] [input_dir] [virtual_env_with_3to2]
```

If you don't specify any arguments then the input directory will be the current
working directory, the output directory will be `python2.7-output` within the
current directory, and `3to2` will be invoked in the current environment.

The script does nothing if the output directory already exists. 


### Producing a pypi package

1. Set the version numbers in [cirq/_version.py](/cirq/_version.py).

    The python 3 version should end with `.35` whereas the python 2 version should end with `.27`.
    For example: `0.0.4.35` is the python 3 variant of version `0.0.4`.
    `pip` will choose between the two based on whichever version of python the user is using.
    Development versions end with `.dev35` and `.dev27` instead of `.35` and `.27`, e.g. use `0.0.4.dev27` for the python 2 variant of the development version of `0.0.4`.

2. Run [dev_tools/prepare-package.sh](/dev_tools/produce-package.sh) to produce pypi artifacts.

    ```bash
    bash dev_tools/produce-package.sh
    ```

    The output files will be placed in `dist/`, from which they can be uploaded to pypi with a tool such as `twine`.
