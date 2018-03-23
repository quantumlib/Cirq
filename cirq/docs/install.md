## Installation

### Summary

For the experts, installation is as simple as
```bash
pip install git+https://git@github.com/quantumlib/cirq.git#egg=cirq
```
You will be asked for credentials: use an access token for your github
account generated at
[https://github.com/settings/tokens](https://github.com/settings/tokens).
Alternative if you have github set up for ssh on your machine, 
you can use
```bash
pip install git+ssh://git@github.com/quantumlib/cirq.git#egg=cirq
``` 

### Detailed Install on Mac or Ubuntu 

We recommend using [Virtualenv](https://virtualenv.pypa.io/en/stable/) when
working with Cirq.  Vitualenv creates a Python environment isolated from 
other Python installations. This isolation ensures that other Python 
environment's dependencies or configurations do not interfere with the
environment you have created to work with Cirq.

**Prerequisites:**
 - Python 3 and Pip. See these 
[instructions](http://docs.python-guide.org/en/latest/starting/installation/)
for installing both of these. 

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

5. Pip install Cirq and its dependencies. From the directory where 
you cloned Cirq:
```bash
(targetDirectory)$ pip3 install git+https://git@github.com/quantumlib/cirq.git#egg=cirq
``` 
You will be asked for credentials: use an access token generated at
[https://github.com/settings/tokens](https://github.com/settings/tokens) 
with the "repo" permissions enabled.

If you want to install Cirq and change Cirq code, then you should add the
```-e``` flag to the above command:
```bash
(targetDirectory)$ pip3 install -e git+https://git@github.com/quantumlib/cirq.git#egg=cirq
```  
Note that Cirq will be copied into the ```src```directory of your 
virtual environment, so use this as the base of 
your git repo. For contributing to Cirq you should use git and
follow these [guidelines](../../CONTRIBUTING.md). See these 
[instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
for installing git.  

6. You should now be able use Cirq. To confirm this you should be able
to run the following with no errors
```bash
(targetDirectory)$ python   # Or your command to run python.
>>> import cirq
```

7. You can leave the virtual environment by typing ```deactivate```
at any time. To re-enter this environment follow the instructions in 
step 4.   

**Matplotlib note**

If you get an ```ImportError: No module named 'tkinter'``` on ```import matplotlib.pyplot```,
install python3-tk via ```apt``` instead of ```pip3```:

```bash
$ sudo apt-get install python3-tk
```

### Install on Windows

TODO
