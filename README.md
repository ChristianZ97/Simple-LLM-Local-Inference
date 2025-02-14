# Simple LLM Local Inference Setup on WSL

This guide outlines the steps to set up a local LLM inference environment using WSL.

## 1. Reset and Check WSL
```bash
wsl --unregister Ubuntu
wsl --list --verbose
```

## 2. Check the Installation of Git and Python
```bash
git --version
python3 --version
# If not installed:
sudo apt install git
sudo apt install python3
```

## 3. Install NVIDIA CUDA Toolkit
Check GPU tools:
```bash
nvidia-smi
nvcc --version
```

If `nvcc` is not installed, run (please follow the [official guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)):
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

## 4. Set Up Python Virtual Environment
Check pip:
```bash
pip --version
# If not installed:
sudo apt install python3-pip
```

Install the venv module and create a virtual environment:
```bash
sudo apt install python3.12-venv
python3 -m venv local_llm_test
source local_llm_test/bin/activate
```

## 5. Clone the Project
```bash
git clone https://github.com/ChristianZ97/Simple-LLM-Local-Inference
cd Simple-LLM-Local-Inference/
```

Install required Python packages:
```bash
pip install -r requirements.txt
```

## 6. Log in to Hugging Face
```bash
huggingface-cli login
```

## 7. Run the Project
```bash
python3 local_inference_llama.py
```

## License
This project is licensed under the MIT License.
