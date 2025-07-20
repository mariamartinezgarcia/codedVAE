import torch
from torchvision import datasets, transforms
import src.nn.modules as mod
from src.coded_vae import CodedVAE
import pickle
import os
import random
import numpy as np
from torch.utils.data import DataLoader
import yaml
import argparse

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def cyclic_diagonal_matrix(rows, cols):
    """
    Generates a matrix where diagonals wrap around when the number of columns 
    exceeds the number of rows.
    
    Args:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
    
    Returns:
        np.ndarray: The generated matrix.
    """
    # Initialize an empty matrix
    matrix = torch.zeros((rows, cols))

    # Fill diagonals cyclically
    for i in range(cols):
        row_index = i % rows  # Wrap around when reaching last row
        matrix[row_index, i] = 1.  # Assign 1 to diagonal positions

    return matrix

g = torch.Generator()
g.manual_seed(0)

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config_file", help="configuration file", type=str, default='config_train_hierarchical.yml')
    args = parser.parse_args()
    config_file = args.config_file

    # ---- Load config ---- #
    with open('./configs/'+config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f"Loaded configuration file {config_file}")

    # GPU
    use_cuda =  config["train_config"]["use_cuda"] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device= torch.device("cpu")

    # Inference type
    inf_type = config['model_config']['inf_type']
    likelihood = config['model_config']['likelihood']

    # Load matrices
    polar = config['model_config']['polar']
    bits_info = config['model_config']['bits_info'] #list
    bits_code = config['model_config']['bits_code'] #list
    code_path = []
    if config['model_config']['code_file'] == 'default':
        for i in range(len(config['model_config']['bits_info'])):
            code_path.append('./codes/rep_matrices_'+str(bits_info[i])+'_'+str(bits_code[i])+'.pkl')
    else:
        for i in range(len(config['model_config']['bits_info'])):
            code_path.append('./codes/'+config['model_config']['code_file'][i])
    
    rep_matrices = []
    for i in range(len(code_path)):
        with open(code_path[i], 'rb') as file:
            rep_matrices.append(pickle.load(file))

    G_list = []
    for i in range(len(rep_matrices)):
        G_list.append(rep_matrices[i]['G'])

    # Adjacency matrix. 
    # Each matrix A idicate which information bits should  be combined. Sorted in ascenting order (A[0] indicates combination m_1+m_2, A[1] indicates combination m_1+m_2, and so on)
    A = []
    for i in range(1,len(bits_info)):
        A.append(cyclic_diagonal_matrix(bits_info[i-1], bits_info[i]))

    
    bits_info_total = sum(bits_info)
    bits_code_total = sum(bits_code)

    # ---- Load data ---- #
    os.makedirs("data/", exist_ok=True)

    dataset = config['train_config']['dataset'] # Dataset can be 'MNIST', 'FMNIST', 'CIFAR10', or 'IMAGENET' 
    batch_size = config["train_config"]["batch_size"]

    # MNIST # 
    if dataset == 'MNIST':

        D = 28*28
        # Download and load the training data
        trainset = datasets.MNIST('./data/', download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]) )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # whole dataset

    # FMNIST #
    if dataset == 'FMNIST':

        D = 28*28
        # Download and load the training data
        trainset = datasets.FashionMNIST('./data/FMNIST/', download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    if dataset == 'CIFAR10':

        D=32*32*3
        # Download and load the training data
        trainset = datasets.CIFAR10(root='./data/CIFAR10/',download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    if dataset == 'IMAGENET':

        D=64*64*3
        # Download and load the training data
        trainset = datasets.ImageFolder('./data/tiny-imagenet-200/train', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # ---- Get encoder and decoder networks---- #
    enc = mod.get_encoder(config['model_config']['type_encoder'], bits_code_total, dataset)
    dec = mod.get_decoder(config['model_config']['type_decoder'], bits_code_total, dataset)
    
    # ---- Declare model ---- #
    model = CodedVAE(enc, dec, bits_code_total, bits_info=bits_info, likelihood=likelihood, G=G_list, A=A, beta=config['model_config']['beta'], lr=config['train_config']['lr'], inference=inf_type, seed=0, device=device, polar=polar)
    model.to(device)

    # ---- Train model ---- #
    elbo_evol, kl_evol, rec_evol = model.train(trainloader, n_epochs=config['train_config']['n_epochs'])

    # ---- Save model ---- #
    if config['train_config']['save_model']:
        os.makedirs("checkpoints/", exist_ok=True)
        if config['train_config']['checkpoint'] == 'default':
            model.save('./checkpoints/hier_'+dataset+'_'+str(bits_info)+'_'+str(bits_code)+'_checkpoint.pt')
        else:
            model.save('./checkpoints/'+config['train_config']['checkpoint'])

if __name__ == '__main__':
    main()
