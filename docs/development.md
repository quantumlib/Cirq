## Development

This document is a summary of how to do various tasks one runs into as a
developer of cirq.
Note that all commands assume a Debian environment, and all commands (except
the initial repository cloning command) assume you are at the cirq repo root.


### Setting up an environment.

0. Clone the repository.

    ```bash
    git clone git@github.com:quantumlib/Cirq.git
    cd Cirq
    ```

1. Install system dependencies.

    Make sure you have python 3.5 or greater.
    You can install most other dependencies via `apt-get`:

    ```bash
    cat apt-dev-requirements.txt apt-runtime-requirements.txt | xargs sudo apt-get install --yes
    ```

    Unfortunately, as of this writing, v3.5 of the [protobuf compiler](https://github.com/google/protobuf) is required but not installable via `apt-get`.
    Without this dependency, you will not be able to produce the transpiled python 2.7 code.

    The simplest way to install v3.5 of the protobuf compiler dependency is to download
    `https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip`,
    unzip it into some directory of your choice,
    and add that directory to your path.

    For example, that is how the travis continuous integration scripts get the protobuf compiler:

    ```bash
    # Grab precompiled protobuf compiler v3.5.1
    curl -OL https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip
    # Drop directly in current working directory.
    unzip protoc-3.5.1-linux-x86_64.zip -d protoc3.5
    PATH=$(pwd)/protoc3.5/bin:${PATH}
    ```

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

There are two options, one easy/fast and the other slow/reliable.

Run [continuous-integration/simple_check.sh](/continuous-integration/simple_check.sh) to invoke `pylint`, `mypy`, and `pytest` directly on your working directory.
This check does not attempt to ensure your test environment is up to date, and it does not transpile and test the python 2 code.

```bash
bash continuous-integration/simple_check.sh
```

Run [continuous-integration/check.sh](/continuous-integration/check.sh) to run the checks inside a temporary fresh virtual environment and to also transpile and test the python 2 code. 

```bash
bash continuous-integration/check.sh
```


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
