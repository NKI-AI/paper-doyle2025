import os
import subprocess
import dotenv

def setup_environment():
    """Set up environment variables for the project."""
    # Enable full error traces in Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # Determine base path and source path for the .env file
    base_path = os.getcwd().split("DROP")[0]
    if base_path == "/home/user/project/":
        source_path = f"{base_path}/DROP/env_server.txt"
    else:
        source_path = f"{base_path}/DROP/env_ubuntu.txt"

    # Set destination path for the .env file
    dest_path = f"{base_path}/DROP/.env"

    # Copy the appropriate .env file to the destination
    subprocess.run(["cp", source_path, dest_path], check=True)

    # Load environment variables from the .env file
    dotenv.load_dotenv(dest_path, override=True)

if __name__ == "__main__":
    setup_environment()
