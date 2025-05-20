# Install

Choose your operating system:

- [Installing on Linux](#installing-on-linux)
- [Installing on MacOS](#installing-on-macos)
- [Installing on Windows](#installing-on-windows)

If you want to create a development environment, see the [development page](../dev/development.md).

---

## Python version support

Cirq currently supports Python 3.11 and later.
We follow NumPy's schedule for Python version support defined in [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html),
though we may deviate from that schedule by extending support for older Python
versions if they are needed by [Colab](https://colab.research.google.com/)
or internal Google systems.

## Installing on Linux

0. Make sure you have Python 3.11.0 or greater.

    See [Installing Python 3 on Linux](https://docs.python-guide.org/starting/install3/linux/) in The Hitchhiker's Guide to Python.

1. Consider using a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

2. Use `pip` to install `cirq`:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install cirq
    ```

3. (Optional) install other dependencies.

    Install dependencies of features in `cirq.contrib`.

    ```bash
    python -m pip install 'cirq-core[contrib]'
    ```

    Install system dependencies that `pip` can't handle, such as `texlive-latex-base` to support
    PDF printing. For Debian-based Linux systems, the necessary packages can be installed using
    the following command from the top level of the Cirq repository:

    ```bash
    sudo apt install $(cat apt-system-requirements.txt)
    ```

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


## Installing on MacOS

0. Make sure you have Python 3.11.0 or greater.

    See [Installing Python 3 on MacOS](https://docs.python-guide.org/starting/install3/osx/) .

1. Consider using a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

2. Use `pip` to install `cirq`:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install cirq
    ```

3. (Optional) install dependencies of features in `cirq.contrib`.

    ```bash
    python -m pip install 'cirq-core[contrib]'
    ```

    Install system dependencies that `pip` can't handle.

    ```bash
    brew install --cask mactex
    ```

    - Without `mactex`, functionality for writing PDF files will not work.

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

1. Make sure you have Python 3.11.0 or greater.

    See [Installing Python 3 on Windows](https://docs.python-guide.org/starting/install3/win/) in The Hitchhiker's Guide to Python.

2. Use `pip` to install `cirq`:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install cirq
    ```

3. (Optional) install dependencies of features in `cirq.contrib`.

    ```bash
    python -m pip install 'cirq-core[contrib]'
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
