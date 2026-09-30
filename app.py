import subprocess
import sys
import os

# Get the directory of the current script (launch_app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to src/main.py
main_script_path = os.path.join(current_dir, 'src', 'main.py')

# Activate the virtual environment
# This assumes the venv is named 'venv' and is in the root directory
if sys.platform == "win32":
    venv_python = os.path.join(current_dir, 'venv', 'Scripts', 'python.exe')
else: # Unix-like systems (Linux, macOS)
    venv_python = os.path.join(current_dir, 'venv', 'bin', 'python')

if not os.path.exists(venv_python):
    print(f"Error: Virtual environment Python interpreter not found at {venv_python}")
    print("Please ensure you have created and activated your virtual environment correctly.")
    sys.exit(1)

# Launch the main application using the venv's python interpreter
try:
    print(f"Launching {main_script_path} using {venv_python}...")
    subprocess.run([venv_python, main_script_path], stdout=sys.stdout, stderr=sys.stderr)
except Exception as e:
    print(f"An error occurred while launching the application: {e}")
    sys.exit(1)