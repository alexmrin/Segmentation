import torch

learning_rate = 1e-2
num_epochs = 250
momentum = 0.9
weight_decay = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
num_workers = 1
save_path = './saves'
data_path = './data'
image_dimension = 256
