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

g = torch.Generator()
g.manual_seed(0)

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config_file", help="configuration file", type=str, default='config_train_coded.yml')
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

    # Code
    bits_info = config['model_config']['bits_info']
    bits_code = config['model_config']['bits_code']

    if config['model_config']['code_file'] == 'default':
        code_path = './codes/rep_matrices_'+str(bits_info)+'_'+str(bits_code)+'.pkl'
    else:
        code_path = './codes/'+config['model_config']['code_file']

    with open(code_path, 'rb') as file:
        rep_matrices = pickle.load(file)

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
    enc = mod.get_encoder(config['model_config']['type_encoder'], bits_code, dataset)
    dec = mod.get_decoder(config['model_config']['type_decoder'], bits_code, dataset)

    # ---- Declare model ---- #
    model = CodedVAE(enc, dec, bits_code, likelihood=likelihood, G=rep_matrices['G'], beta=config['model_config']['beta'], lr=config['train_config']['lr'], inference=inf_type, seed=0, device=device)
    model.to(device)

    # ---- Train model ---- #
    elbo_evol, kl_evol, rec_evol = model.train(trainloader, n_epochs=config['train_config']['n_epochs'])

    # ---- Save model ---- #
    if config['train_config']['save_model']:
        os.makedirs("checkpoints/", exist_ok=True)
        if config['train_config']['checkpoint'] == 'default':
            model.save('./checkpoints/coded_'+dataset+'_'+str(bits_info)+'_'+str(bits_code)+'_checkpoint.pt')
        else:
            model.save('./checkpoints/'+config['train_config']['checkpoint'])

if __name__ == '__main__':
    main()
