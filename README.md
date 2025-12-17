
# Adaptive Screen AI

Desktop dashboard for adaptive screen comfort (PySide6 + OpenCV + MediaPipe). Requires a webcam and Windows brightness control for full functionality.

## Quick start
1) Create the PySide6-focused virtual env (name matches the working command):
	- `python -m venv .pyside-env`

2) Install dependencies inside that env:
	- `./.pyside-env/Scripts/python.exe -m pip install -r requirements.txt`

3) Run the app (command confirmed working):
	- `./.pyside-env/Scripts/python.exe app.py`

If you prefer an existing env, be sure its `PATH` resolves the PySide6 DLLs; otherwise stick with `.pyside-env` to avoid Qt plugin/DLL issues.

