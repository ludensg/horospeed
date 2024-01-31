
# Script to set up the virtual environment for Horovod and DeepSpeed experiment

# Update package manager and install python3-venv if needed
#sudo apt update
#sudo apt install python3.10-venv

# Create a new virtual environment
python3 -m venv horospeed-env

# Activate the virtual environment
source horospeed-env/bin/activate

# Install PyTorch
pip install torch torchvision

# Install Horovod with PyTorch support
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]

# Install DeepSpeed
pip install deepspeed
