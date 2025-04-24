### Python 3.13 users

Braindecode has not bumped its `python_requires` yet.  
Before installing, export one env-var so `pip` ignores that guard:

```powershell
set PIP_IGNORE_REQUIRES_PYTHON=1   # (Linux/macOS:  export PIP_IGNORE_REQUIRES_PYTHON=1)
pip install --no-build-isolation -r requirements.txt
remove-item env:PIP_IGNORE_REQUIRES_PYTHON  # (Linux/macOS:  unset ...)
