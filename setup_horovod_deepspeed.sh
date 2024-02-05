
# Script to set up the virtual environment for Horovod and DeepSpeed experiment

# Update package manager and install python3-venv if needed
#sudo apt update
#sudo apt install python3.10-venv

# Check if the virtual environment already exists
if [ ! -d "horospeed-env" ]; then
    # Create a new virtual environment if it doesn't exist
    python3 -m venv horospeed-env
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Attempt to activate the virtual environment
source horospeed-env/bin/activate

# Check if the virtual environment was activated successfully
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "Virtual environment activated."

    # Install PyTorch
    pip install torch torchvision

    # Install Horovod with PyTorch support
    HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]

    # Install DeepSpeed
    pip install deepspeed

    echo "Dependencies installed."
else
    echo "Failed to activate virtual environment."
fi