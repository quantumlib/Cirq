# Install

Choose your operating system:

- [Installing on Linux](#installing-on-linux)
- [Installing on Mac OS X](#installing-on-mac-os-x)
- [Installing on Windows](#installing-on-windows)

If you want to create a development environment, see the [development page](/cirq/dev/development).

---

## Python version supprt

Cirq currently supports python 3.7 and later.
We follow numpy's schedule for python version support defined in [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html),
though we may deviate from that schedule by extending support for older python
versions if they are needed by [Colab](https://colab.research.google.com/)
or internal Google systems.

## Installing on Linux

0. Make sure you have python 3.7.0 or greater.

    See [Installing Python 3 on Linux](https://docs.python-guide.org/starting/install3/linux/) @ the hitchhiker's guide to python.

1. Consider using a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

2. Use `pip` to install `cirq`:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install cirq
    ```

3. (Optional) install other dependencies.

    Install dependencies of features in `cirq.contrib`.

    ```bash
    python -m pip install cirq-core[contrib]
    ```

    Install system dependencies that pip can't handle.

    ```bash
    sudo apt-get install texlive-latex-base latexmk
    ```

    - Without `texlive-latex-base` and `latexmk`, pdf writing functionality will not work.

4. Check that it works!

    ```bash
    python -c 'import cirq_google; print(cirq_google.Sycamore)'
    # should print:
    #                                              (0, 5)───(0, 6)
    #                                              │        │
    #                                              │        │
    #                                     (1, 4)───(1, 5)───(1, 6)───(1, 7)
    #                                     │        │        │        │
    #                                     │        │        │        │
    #                            (2, 3)───(2, 4)───(2, 5)───(2, 6)───(2, 7)───(2, 8)
    #                            │        │        │        │        │        │
    #                            │        │        │        │        │        │
    #                   (3, 2)───(3, 3)───(3, 4)───(3, 5)───(3, 6)───(3, 7)───(3, 8)───(3, 9)
    #                   │        │        │        │        │        │        │        │
    #                   │        │        │        │        │        │        │        │
    #          (4, 1)───(4, 2)───(4, 3)───(4, 4)───(4, 5)───(4, 6)───(4, 7)───(4, 8)───(4, 9)
    #          │        │        │        │        │        │        │        │
    #          │        │        │        │        │        │        │        │
    # (5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)───(5, 5)───(5, 6)───(5, 7)───(5, 8)
    #          │        │        │        │        │        │        │
    #          │        │        │        │        │        │        │
    #          (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)───(6, 6)───(6, 7)
    #                   │        │        │        │        │
    #                   │        │        │        │        │
    #                   (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
    #                            │        │        │
    #                            │        │        │
    #                            (8, 3)───(8, 4)───(8, 5)
    #                                     │
    #                                     │
    #                                     (9, 4)
    ```


## Installing on Mac OS X

0. Make sure you have python 3.7.0 or greater.

    See [Installing Python 3 on Mac OS X](https://docs.python-guide.org/starting/install3/osx/) @ the hitchhiker's guide to python.

1. Consider using a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

2. Use `pip` to install `cirq`:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install cirq
    ```

3. (Optional) install dependencies of features in `cirq.contrib`.

    ```bash
    python -m pip install cirq-core[contrib]
    ```

    Install system dependencies that pip can't handle.

    ```bash
    brew cask install mactex
    ```

    - Without `mactex`, pdf writing functionality will not work.

4. Check that it works!

    ```bash
    python -c 'import cirq_google; print(cirq_google.Sycamore)'
    # should print:
    #                                              (0, 5)───(0, 6)
    #                                              │        │
    #                                              │        │
    #                                     (1, 4)───(1, 5)───(1, 6)───(1, 7)
    #                                     │        │        │        │
    #                                     │        │        │        │
    #                            (2, 3)───(2, 4)───(2, 5)───(2, 6)───(2, 7)───(2, 8)
    #                            │        │        │        │        │        │
    #                            │        │        │        │        │        │
    #                   (3, 2)───(3, 3)───(3, 4)───(3, 5)───(3, 6)───(3, 7)───(3, 8)───(3, 9)
    #                   │        │        │        │        │        │        │        │
    #                   │        │        │        │        │        │        │        │
    #          (4, 1)───(4, 2)───(4, 3)───(4, 4)───(4, 5)───(4, 6)───(4, 7)───(4, 8)───(4, 9)
    #          │        │        │        │        │        │        │        │
    #          │        │        │        │        │        │        │        │
    # (5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)───(5, 5)───(5, 6)───(5, 7)───(5, 8)
    #          │        │        │        │        │        │        │
    #          │        │        │        │        │        │        │
    #          (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)───(6, 6)───(6, 7)
    #                   │        │        │        │        │
    #                   │        │        │        │        │
    #                   (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
    #                            │        │        │
    #                            │        │        │
    #                            (8, 3)───(8, 4)───(8, 5)
    #                                     │
    #                                     │
    #                                     (9, 4)
    ```


## Installing on Windows

0. If you are using the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about), use the [Linux install instructions](#installing-on-linux) instead of these instructions.

1. Make sure you have python 3.7.0 or greater.

    See [Installing Python 3 on Windows](https://docs.python-guide.org/starting/install3/win/) @ the hitchhiker's guide to python.

2. Use `pip` to install `cirq`:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install cirq
    ```

3. (Optional) install dependencies of features in `cirq.contrib`.

    ```bash
    python -m pip install cirq-core[contrib]
    ```

4. Check that it works!

    ```bash
    python -c "import cirq_google; print(cirq_google.Sycamore)"
    # should print:
    #                                              (0, 5)───(0, 6)
    #                                              │        │
    #                                              │        │
    #                                     (1, 4)───(1, 5)───(1, 6)───(1, 7)
    #                                     │        │        │        │
    #                                     │        │        │        │
    #                            (2, 3)───(2, 4)───(2, 5)───(2, 6)───(2, 7)───(2, 8)
    #                            │        │        │        │        │        │
    #                            │        │        │        │        │        │
    #                   (3, 2)───(3, 3)───(3, 4)───(3, 5)───(3, 6)───(3, 7)───(3, 8)───(3, 9)
    #                   │        │        │        │        │        │        │        │
    #                   │        │        │        │        │        │        │        │
    #          (4, 1)───(4, 2)───(4, 3)───(4, 4)───(4, 5)───(4, 6)───(4, 7)───(4, 8)───(4, 9)
    #          │        │        │        │        │        │        │        │
    #          │        │        │        │        │        │        │        │
    # (5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)───(5, 5)───(5, 6)───(5, 7)───(5, 8)
    #          │        │        │        │        │        │        │
    #          │        │        │        │        │        │        │
    #          (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)───(6, 6)───(6, 7)
    #                   │        │        │        │        │
    #                   │        │        │        │        │
    #                   (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
    #                            │        │        │
    #                            │        │        │
    #                            (8, 3)───(8, 4)───(8, 5)
    #                                     │
    #                                     │
    #                                     (9, 4)
    ```

