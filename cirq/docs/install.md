## Installation

**Note** In its current pre-release state, Cirq can only be installed
from  source. When Cirq is launched publicly, installing using pip will make
this process much easier (essentially it will just be ```pip install cirq```). 
Also note that the instructions below are only for installing Cirq with 
Python 3.x.

### Install on Mac or Ubuntu

We recommend using [Virtualenv](https://virtualenv.pypa.io/en/stable/) when
working with Cirq.  Vitualenv creates a Python environment isolated from 
other Python installations. This isolation ensures that other Python 
environment's dependencies or configurations do not interfere with the
environment you have created to work with Cirq.

**Prerequisites:**
1. Python 3 and Pip. See these 
[instructions](http://docs.python-guide.org/en/latest/starting/installation/)
for installing both of these. 
2. Git. See these
[instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
for installing.  

**Instructions:**
1. Start a shell (terminal). All commands below execute in this shell.
2. Install Virtualenv (if not already installed):
```bash
$ pip install --upgrade virtualenv 
```

3. Create a virtual environment:
```bash
$ virtualenv --no-site-packages -p python3 targetDirectory
```
Here *targetDirectory* is the directory where you want to install the
virtual environment (this does not need to be related to where the code
you write with that uses Cirq lives). It will also be the name of
the virtual environment.

4. Activate the virtualenv:
```bash
$ cd targetDirectory
$ source ./bin/activate      # If using bash, sh, ksh, or zsh
$ source ./bin/activate.csh  # If using csh or tcsh 
```
After this your shell should show that you are in this virtual environment
by appending the target directory name to the shell prompt:
```bash
(targetDirectory)$
```

5. Download or clone Cirq from github.  You'll want to decide 
which directory you want to put this.  ```cd``` to that directory and then
clone Cirq:
```bash
(targetDirectory)$ git clone https://github.com/quantumlib/cirq
```
Use your github username to authenticate, and then, because we require
2-factor auth, you will need to use a personal access token as your
password.  This token can be found 
[here](https://github.com/settings/tokens). This will create a directory
called ```cirq``` in which the base of the github repo is installed.

6. Pip install cirq and its dependencies. From the directory where 
you cloned cirq:
```bash
(targetDirectory)$ pip3 install -r cirq/requirements.txt
(targetDirectory)$ pip3 install -e ./cirq 
``` 
If you want to install Cirq without including local edits, remove ```-e```
from the last command. For contributing to Cirq see [here](CONTRIBUTING)

7. You should now be able use Cirq. To confirm this you should be able
to run the following with no errors
```bash
(targetDirectory)$ python   # Or your command to run python.
>>> import cirq
```

8. You can leave the virtual environment by typing ```deactivate```
at any time. To re-enter this environment follow the instructions in 
step 4.   

### Install on Windows

TODO
