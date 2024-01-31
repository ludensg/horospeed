import torch
import torch.nn as nn
import torch.optim as optim
import sys
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV2Processor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'   # Replace with your master node's IP
    os.environ['MASTER_PORT'] = '12355'       # Use a free port

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# ALTERNATIVE FOR SETUP (need to refactor code, use only one)

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])

def run(backend):
    tensor = torch.zeros(1)
    
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)      #   This step is necessary because NCCL requires tensors to be 
                                        #   on the correct GPU device to manage inter-GPU communication effectively.

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('worker_{} has received data from rank {}\n'.format(WORLD_RANK, 0))

def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(backend)


# Command line argument to choose between 'horovod' or 'deepspeed'
framework = sys.argv[1]

# Initialize the tokenizer and model for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Load and preprocess SQuAD dataset
SQUAD_DIR = "./squad" # directory for local SQuAD setup

processor = SquadV2Processor()
train_examples = processor.get_train_examples(SQUAD_DIR, filename='train-v2.0.json')
train_features = squad_convert_examples_to_features(examples=train_examples,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=384,
                                                    doc_stride=128,
                                                    max_query_length=64,
                                                    is_training=True)

# Create PyTorch DataLoader
train_dataset = TensorDataset(torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
                              torch.tensor([f.attention_mask for f in train_features], dtype=torch.long),
                              torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long),
                              torch.tensor([f.start_position for f in train_features], dtype=torch.long),
                              torch.tensor([f.end_position for f in train_features], dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load and preprocess SQuAD validation dataset
validation_examples = processor.get_dev_examples(SQUAD_DIR, filename='dev-v2.0.json')
validation_features = squad_convert_examples_to_features(examples=validation_examples,
                                                         tokenizer=tokenizer,
                                                         max_seq_length=384,
                                                         doc_stride=128,
                                                         max_query_length=64,
                                                         is_training=False)

# Create PyTorch DataLoader for validation data
validation_dataset = TensorDataset(torch.tensor([f.input_ids for f in validation_features], dtype=torch.long),
                                   torch.tensor([f.attention_mask for f in validation_features], dtype=torch.long),
                                   torch.tensor([f.token_type_ids for f in validation_features], dtype=torch.long),
                                   torch.tensor([f.start_position for f in validation_features], dtype=torch.long),
                                   torch.tensor([f.end_position for f in validation_features], dtype=torch.long))
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)  # batch size can be adjusted


# Framework-specific setup
if framework == 'horovod':
    import horovod.torch as hvd
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=3e-5 * hvd.size())
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
elif framework == 'deepspeed':
    import deepspeed
    model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(parameters, lr=3e-5)
    model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, model_parameters=parameters)




def main(rank, world_size):
    setup(rank, world_size)

    #
    # TRAINING CODE BEGINS HERE
    #

    num_epochs = 5 # trains for 5 epochs

    # Variables for extra functionality in training
    print_interval = 100  # prints loss every 100 steps
    save_interval = 1000  # saves the model every 1000 steps
    validation_interval = 3  # validates model every 3 epochs
    checkpoint_path = "./checkpoints"

    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, start_positions, end_positions = [b.to(model.device) for b in batch]

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            if framework == 'horovod':
                optimizer.synchronize()
                with optimizer.skip_synchronize():
                    optimizer.step()
            elif framework == 'deepspeed':
                model.backward(loss)
                model.step()

            if i % print_interval == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}")
            
            if i % save_interval == 0:
                model_save_path = f"{checkpoint_path}/model_epoch_{epoch}_step_{i}.pt"
                torch.save(model.state_dict(), model_save_path)

            total_loss += loss.item()
            
        avg_training_loss = total_loss / len(train_loader)
        training_losses.append(avg_training_loss)

        # Perform validation at specified intervals
        if (epoch + 1) % validation_interval == 0:
            validation_loss = evaluate_model(model, validation_loader)
            validation_losses.append(validation_loss)
            print(f"Validation Loss after Epoch {epoch}: {validation_loss}")

    # Training loop is over
    plot_model_performance(training_losses, validation_losses, validation_interval)


    #
    # TRAINING CODE OVER
    #
    cleanup()

if __name__ == "__main__":
    world_size = 2  # Number of nodes
    rank = int(os.environ['RANK'])  # Rank will be provided by the job scheduler
    main(rank, world_size)


def evaluate_model(model, validation_loader):
    model.eval()
    total_loss, total_steps = 0, 0

    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, token_type_ids, start_positions, end_positions = [b.to(model.device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    return avg_loss


def plot_model_performance(training_losses, validation_losses, validation_interval):
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(range(validation_interval - 1, len(training_losses), validation_interval), validation_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Performance: Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

