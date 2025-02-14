# Simple LLM Local Inference

## Setup on WSL (Ubuntu)

- **Reset and Check WSL, Git and Python**
  ```bash
  wsl --unregister Ubuntu
  wsl --list --oneline
  wsl --install -d Ubuntu

  git --version
  python3 --version
  
  # If not installed:
  sudo apt install git
  sudo apt install python3
  ```
  
- **Install NVIDIA CUDA Toolkit**
  ```bash
  # Check GPU tools
  nvidia-smi
  nvcc --version
  ```

- If `nvcc` is not installed, run (please follow the [official guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)):
  ```bash
  sudo apt-key del 7fa2af80
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
  sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.0-1_amd64.deb
  sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cuda-toolkit-12-8
  
  sudo apt install nvidia-cuda-toolkit
  ```

- **Set Up Python Virtual Environment**

- Install the venv module and create a virtual environment
  ```bash
  sudo apt install python3.12-venv
  python3 -m venv local_llm_test
  source local_llm_test/bin/activate
  ```
  
- Install pip if needed
  ```bash
  pip --version
  # If not installed:
  sudo apt install python3-pip
  ```

- **Run the Project**
  ```bash
  # Clone the project
  git clone https://github.com/ChristianZ97/Simple-LLM-Local-Inference
  cd Simple-LLM-Local-Inference/
  
  # Install required packages
  pip install -r requirements.txt
  
  # Log in to Hugging Face
  huggingface-cli login
  
  # Run local inference
  python3 local_inference_llama.py
  ```

## Setup Docker for Testing Inference on NVIDIA Edge Device

- **Download Docker Desktop**  
  Visit [Docker Desktop](https://www.docker.com/) and download the installer.
  ```bash
  docker --version
  ```

- **Running NVIDIA L4T JetPack Containers**
  ```bash
  # Copy the project's script to the docker workspace
  docker run --rm -it -v /path/to/Simple-LLM-Local-Inference:/workspace nvcr.io/nvidia/l4t-jetpack:r36.4.0 /bin/bash
  ```

- **Run the project**
  ```bash
  cd /workspace
  pip install -r requirements.txt
  python3 local_inference_llama.py
  ```

---

## License
This project is licensed under the MIT License.
