# Distributed Deep Learning Experiment with BERT and SQuAD

## Overview

This experiment is designed to compare the performance of two distributed deep learning frameworks, Horovod and DeepSpeed, using BERT model training on the SQuAD dataset, and NCCL for GPU-to-GPU communication.

### Rule of thumb

- Use the NCCL backend for distributed GPU training
- Use the Gloo backend for distributed CPU training.

### What is SQuAD?
The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset consisting of questions posed upon a set of Wikipedia articles, where the answers to each question are segments of text from the corresponding articles.

### What is BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing pre-training. It's designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context.

### What is Horovod?
Horovod is an open-source distributed deep learning training framework, created by Uber. It's used to improve the speed, scale, and resource allocation in machine learning training activities. Horovod is known for its easy usability and efficient distributed training capabilities, especially in environments with multiple GPUs.

### What is DeepSpeed?
DeepSpeed is a deep learning optimization library that provides a suite of tools to train large models and datasets efficiently. Developed by Microsoft, it's particularly known for enabling training of massive models (with billions of parameters) efficiently and with a lower computational resource footprint.

## File Descriptions

1. **`training.py`**: This is the main Python script that trains the BERT model on the SQuAD dataset. It supports distributed training using either Horovod or DeepSpeed, as specified by the user.

2. **`prepare4training.sh`**: A Bash script to prepare the environment for running `training.py`. It ensures that the necessary virtual environment is activated, and required directories (`checkpoints` and `squad`) exist.

3. **`setup_horovod_deepspeed.sh`**: A Bash script to install necessary dependencies. Mainly the creation of a virtual environment, and installation of Python, PyTorch, Horovod (with Python support), and DeepSpeed

## `training.py` Procedures

### Preparing the Data and Model
- The script starts by importing necessary libraries including PyTorch, transformers, and the chosen distributed training framework (Horovod or DeepSpeed).
- The BERT tokenizer and model for question answering are initialized.
- The SQuAD dataset is loaded and preprocessed into a format suitable for training with BERT.
- A PyTorch DataLoader is created for handling the dataset during training.

### Setting Up Distributed Training
- Depending on the command-line argument, the script configures either Horovod or DeepSpeed for distributed training.
- In the case of Horovod, it initializes Horovod, sets the appropriate GPU device, and scales the learning rate.
- For DeepSpeed, it sets up the model and optimizer using the DeepSpeed engine.

### Training Loop
- The model is trained over a specified number of epochs. In each epoch, the script processes batches of data, performs forward and backward passes, and updates the model parameters.
- Loss is calculated for each batch, and the script optionally prints this loss at specified intervals for monitoring.
- The model is saved at regular intervals to ensure that training progress is not lost.

### Validation and Performance Tracking
- The script evaluates the model on a validation set at specified epoch intervals to monitor its performance on unseen data.
- Training and validation loss are tracked and plotted to visualize the model's performance over time.

### Additional Functionalities
- The script includes additional functions for evaluating the model and plotting its performance.
- These include a function to calculate the average loss over the validation dataset and a function to plot training and validation losses over epochs.

---

## Running the Experiment
To run the experiment, first (if necessary) execute `prepare4training.sh` to set up the environment, and then run `training.py` with the desired framework (Horovod or DeepSpeed) as an argument.

## Future Work
Future work at the moment will be:
- The implementation of the support for more arguments for manipulating variables for experimentation and performance comparison in scaling, mainly:
    - Number of GPUs to be used
    - Batch size
    - Learning rate
- The implementation of the option to manipulate training variables through arguments, mainly:
    - Amount of epochs
    - Intervals for printing, validating, and saving/checkpointing.
- The implementation of a way of monitoring GPU and memory usage to analyze resource utilization. 
- Thorough testing, to produce experimental results in a future report.

## Potential Ways Forward
Either focusing on just GPU, or also analyze CPU performance, or even a mixture of CPU to GPU communication and computation.