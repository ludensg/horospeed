#!/bin/bash

# Define the path to the virtual environment and dataset
ENV_PATH="./horospeed-env"
SQUAD_DIR="./squad"
CHECKPOINTS_DIR="./checkpoints"

# Activate the virtual environment
if [ -d "$ENV_PATH" ]; then
    echo "Activating virtual environment..."
    source $ENV_PATH/bin/activate
else
    echo "Virtual environment not found. Please create it first."
    exit 1
fi

# Check and create SQuAD directory if it doesn't exist
if [ ! -d "$SQUAD_DIR" ]; then
    echo "Creating SQuAD directory..."
    mkdir $SQUAD_DIR
    # Optional: Maybe add commands to download SQuAD dataset here
fi

# Check and create checkpoints directory if it doesn't exist
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Creating checkpoints directory..."
    mkdir $CHECKPOINTS_DIR
fi

# Now the environment is set up, you can run the training script
# python training.py <args> (horovod or deepspeed)
