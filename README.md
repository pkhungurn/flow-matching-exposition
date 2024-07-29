# flow-matching-exposition

An exposition on flow matching.

## System setup for local development

1. [pyenv-win](https://github.com/pyenv-win/pyenv-win)

   Follow the [installation instruction](https://github.com/pyenv-win/pyenv-win#quick-start) on the repository.

   Note that you might not be able to execute the command because of the execution policy of PowerShell. See [this link](https://www.mtioutput.com/entry/2020/12/01/131950) for how to circumvent it.

   After installing pyenv-win, install a global version of Python, and you should be done for now.

2. [Poetry](https://python-poetry.org/)

   Follow the guide [here](https://python-poetry.org/docs/#installation) to install. For Windows, use PowerShell.

### Creating Python development environment for the first time in Windows

Do this every time you clone the repository to a new Windows computer.

First, if you have not done so, use `pyenv` to install Python version 3.10.11.

```
pyenv install 3.10.11
```

Now, create a virtual environment called `windows` under the directory python/venvs

```
cd <repo-root>/python/venvs
python -m venv windows --prompt flow-matching-exposition
```

To activate the environment, run its activation script.

```
cd <repo-root>/flow-matching-exposition
python/venvs/windows/Scripts/activate.bat
```

Then, use Poetry to install dependencies.

```
cd <repo-root>/python/poetries/windows
poetry install --no-root
```