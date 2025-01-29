import os
import subprocess
import venv

def create_venv(venv_path="venv"):
  """Creates a virtual environment if it doesn't exist and installs dependencies."""
  if not os.path.exists(venv_path):
    print("Creating virtual environment...")
    venv.create(venv_path, with_pip=True)
  
  # Determine the path to the python executable inside the venv
  if os.name == "nt":
    python_executable = os.path.join(venv_path, "Scripts", "python.exe")
  else:
    python_executable = os.path.join(venv_path, "bin", "python")
  
  print("Upgrading pip...")
  subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
  
  print("Installing dependencies...")
  subprocess.check_call([python_executable, "-m", "pip", "install", "-r", "requirements.txt"])
  print("Done!")

if __name__ == "__main__":
  create_venv()
