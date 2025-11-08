This guide assumes you are on a Linux-based system with an NVIDIA GPU.
Quick Start
Prerequisites
Before you begin, ensure you have the following installed on your system:
git
python3.11 (The setup script specifically requires this version)
curl and wget
An NVIDIA GPU with the appropriate CUDA drivers
1. Clone the Repository
First, clone this repository and move into the project directory:
git clone [https://github.com/Harshkaushik04/Sa2VA](https://github.com/Harshkaushik04/Sa2VA)
cd Sa2VA


2. Run the Setup Script
The req.sh script will automate the entire setup process. Make it executable and run it.
chmod +x req.sh
./req.sh


This script will perform the following steps:
Install uv: Downloads and installs the fast Python package manager.
Create Virtual Environment: Creates a local ./venv folder with python3.11.
Install Dependencies: Uses uv to install all required packages (PyTorch, Transformers, etc.) from pyproject.toml.
Download Models: Downloads the SAM2-Hiera and InternVL3-2B models into the pretrained/ folder.
Download & Process Data: Downloads the Ref-COCO dataset from Hugging Face and runs the convert_to_format.py script to prepare it for training.
This step will take a significant amount of time, as it downloads several gigabytes of models and data.
3. Start Fine-Tuning
Once the setup is complete, start the training process:

bash tools/dist.sh train projects/sa2va/configs/my_finetune.py 1


Configuration
GPU Count
The final command is configured to run on a single GPU.
# The '1' at the end specifies the number of GPUs
bash tools/dist.sh train projects/sa2va/configs/my_finetune.py 1


If you are running on a machine with 8 GPUs, for example, simply change the number:
bash tools/dist.sh train projects/sa2va/configs/my_finetune.py 8


Training Parameters
All other training parameters (batch size, learning rate, max epochs, etc.) are defined in the config file: projects/sa2va/configs/my_finetune.py.
