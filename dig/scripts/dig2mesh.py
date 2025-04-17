import subprocess
import os
from pathlib import Path
import tyro

# Running this script assumes that you are working within the xi workspace with SuGaR conda environment installed
# This script will run the SuGaR train pipeline and extract the mesh from the saved dig state

def stream_output(process):
    """Stream the output from the process while it's running"""
    for line in iter(process.stdout.readline, b''):
        print(line.decode('utf-8').strip())

def run_script_in_env(env_name, script_path, *args):
    # Get the path to the conda environment's Python executable
    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"Current environment: {current_env}")

    shell_command = (
        f"eval \"$(conda shell.bash hook)\" && "
        f"conda activate {env_name} && "
        f"python -u {script_path} {' '.join(str(arg) for arg in args)} && "
        f"conda activate {current_env}"  # Switch back to original environment
    )
    
    my_env = os.environ.copy()
    my_env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        ["bash", "-c", shell_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=my_env
    )
    
    stream_output(process)
    
    return_code = process.wait()
    
    return return_code

def main(
    scene_path: str,
    ):
    
    current_file_path = os.path.abspath(__file__)
    current_directory = Path(os.path.dirname(current_file_path))

    env_name = "sugar"

    script_path = str(current_directory.parent.parent.parent.parent.parent) + "/SuGaR/full_train_pipeline.py"
    args = ["-s", scene_path]
    return_code = run_script_in_env(env_name, script_path, *args)
    print(f"Return code: {return_code}")

if __name__ == "__main__":
    tyro.cli(main)