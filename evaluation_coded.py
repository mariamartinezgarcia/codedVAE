import torch
from torchvision import datasets, transforms
import src.nn.modules as mod
from src.coded_vae import CodedVAE
import pickle
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics
import random

import yaml
import argparse

from torch.utils.data import DataLoader

from src.utils.evaluation import compute_binary_entropy, ClassifierNetwork, eval_reconstruction
from src.utils.sampling import modulate_words

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config_file", help="configuration file", type=str, default='config_eval_coded.yml')
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

    # ---- Obtain Codebook ---- #
    if bits_info<15:
        # Generate all possible words with bits_info
        all_info_words = torch.FloatTensor(list(map(list, itertools.product([0, 1], repeat=bits_info))))
        n_words = all_info_words.shape[0]
        # Encode the info words
        all_words = torch.matmul(all_info_words, rep_matrices['G'])

    # ---- Load data ---- #
    os.makedirs("data/", exist_ok=True)

    dataset = config['eval_config']['dataset'] # Dataset can be 'MNIST', 'FMNIST', 'CIFAR10', or 'IMAGENET' 
    batch_size = config["eval_config"]["batch_size"]

    # MNIST # 
    if dataset == 'MNIST':

        D = 28*28
        # Download and load the test data
        testset = datasets.MNIST('../data/', download=False, train=False, transform=transforms.Compose([transforms.ToTensor()]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        
    # FMNIST #
    if dataset == 'FMNIST':

        D = 28*28
        # Download and load the test data
        testset = datasets.FashionMNIST('./data/FMNIST/', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, generator=g)

    if dataset == 'CIFAR10':

        D=32*32*3
        # Download and load the test data
        testset = datasets.CIFAR10('./data/CIFAR10/', download=True, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

    if dataset == 'IMAGENET':

        D=64*64*3
        # Download and load the test data
        testset = datasets.ImageFolder('./data/tiny-imagenet-200/test', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


    # ---- Get encoder and decoder networks---- #
    enc = mod.get_encoder(config['model_config']['type_encoder'], bits_code, dataset)
    dec = mod.get_decoder(config['model_config']['type_decoder'], bits_code, dataset)

    # ---- Declare model ---- #
    model = CodedVAE(enc, dec, bits_code, likelihood=likelihood, G=rep_matrices['G'], beta=config['model_config']['beta'], lr=config['train_config']['lr'], inference=inf_type, seed=0, device=device)
    model.to(device)

    # ---- Load pre-trained model ---- #
    checkpoint = torch.load('checkpoints/'+config['eval_config']['checkpoint'], map_location=model.device) # Change the path with your own checkpoint!
    model.load_state_dict(checkpoint)
    print('Model loaded!')

    # Evaluation mode
    model.encoder.eval()
    model.decoder.eval()

    # ---- Global entropy ---- #
    if bits_info < 15:
        with torch.no_grad():
            entropy=0
            for images, _ in testloader:
                encoder_out = model.encoder.forward(images.to(model.device))

                logpm1 = torch.matmul(torch.log(encoder_out), model.H.to(model.device))
                logpm0 = torch.matmul(torch.log(1-encoder_out), model.H.to(model.device))

                log_marginals = torch.stack((logpm0, logpm1), dim=2)
                log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

                bit_probs = torch.exp(log_marginals_norm[:,:,1])

                # Compute binary entropy    
                entropy += compute_binary_entropy(bit_probs, all_info_words.to(model.device))

            entropy = entropy/len(testloader)

    print('Global Entropy = ', entropy)

    # ---- WER and BER ---- #
    # 1. Sample random words (uniform distribution over words or uniform distribution for each bit)
    # 2. Forward the sampled words through the decoder to obtain reconstructed images
    # 3. Forward the reconstructed image through the encoder to obtain the recovered word
    # 4. Compare the original sampled word with the recovered word

    with torch.no_grad():
        n_word_samples = 1000

        # 1. Sample random words (uniform distribution for each bit)
        bit_probs = torch.ones((n_word_samples, bits_info))*0.5
        word_info_sample = torch.bernoulli(bit_probs)
        z_sample = modulate_words(word_info_sample, beta=torch.tensor(15.))
        z_sample = torch.matmul(z_sample, rep_matrices['G'])

        # 2. Forward the sampled words through the decoder to obtain reconstructed images
        decoder_out =  model.decoder.forward(z_sample.to(model.device))

        # 3. Forward the reconstructed image through the encoder to obtain the recovered word
        encoder_out = model.encoder.forward(decoder_out)
        logpm1 = torch.matmul(torch.log(encoder_out), model.H.to(model.device))
        logpm0 = torch.matmul(torch.log(1-encoder_out), model.H.to(model.device))

        log_marginals = torch.stack((logpm0, logpm1), dim=2)
        log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)
        word_recovered = torch.bernoulli(torch.exp(log_marginals_norm[:,:,1]))

        # 4. Compare the original sampled word with the recovered word
        ber = (word_info_sample != word_recovered.cpu().data).sum()/(word_info_sample.shape[0]*bits_info)     # Bit Error Rate = n_bit_errors/n_bits_transmitted
        wer = ((word_info_sample != word_recovered.cpu().data).sum(dim=1) != 0).sum()/word_info_sample.shape[0]   # Word Error Rate = n_words_incorrect/n_words_transmitted

        print('BER = ', ber)
        print('WER = ', wer)

    # ---- Generation ---- #
    with torch.no_grad():

        # 1. Generate random images
        generated = model.generate(n_samples=100)
        if dataset=='CIFAR10' or dataset=='IMAGENET':
            generated=generated*0.5 + 0.5
        # 2. Plot generated images
        fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(20,20))
        if dataset=='CIFAR10' or dataset=='IMAGENET':
            for i in range(10):
                for j in range(10):
                    ax = axes[i, j]
                    ax.imshow(np.transpose(generated[i * 10 + j].cpu().data.numpy(), (1,2,0)))  
                    ax.axis('off')
        else:
            for i in range(10):
                for j in range(10):
                    ax = axes[i, j]
                    ax.imshow(generated[i * 10 + j].cpu().data.numpy().reshape([28,28]), cmap='Greys_r')  
                    ax.axis('off')
        plt.tight_layout(pad=0.00)
        plt.show()

    # ---- Reconstruction ---- #
    with torch.no_grad():

        # 1. Obtain a batch of test data
        test = iter(testloader)
        images, labels = next(test)

        # 2. Forward model
        _, reconstructed = model.forward(images)
        # 3. Plot reconstructed images
        if dataset=='CIFAR10' or dataset=='IMAGENET':
            images=images*0.5 + 0.5
            reconstructed=reconstructed*0.5 + 0.5
        fig, axes = plt.subplots(nrows=2, ncols=20, sharex=True, sharey=True, figsize=(40,4))
        for i in range(20):
            if dataset=='CIFAR10'  or dataset=='IMAGENET':
                axes[0,i].imshow(np.transpose(images[i], (1,2,0)))
                axes[0,i].axis('off')
                axes[1,i].imshow(np.transpose(reconstructed[i].cpu().data.numpy(), (1,2,0)))
                axes[1,i].axis('off')
            else:
                axes[0,i].imshow(np.transpose(images[i], (1,2,0)), cmap='Greys_r')
                axes[0,i].axis('off')
                axes[1,i].imshow(np.transpose(reconstructed[i].cpu().data.numpy(), (1,2,0)), cmap='Greys_r')
                axes[1,i].axis('off')
        plt.tight_layout(pad=0.00)
        plt.show()

    # ---- MSE and PSNR Reconstruction ---- #
    with torch.no_grad():
        mse_rec_test = 0
        psnr_test = 0

        for images, _ in testloader:
            _, reconstructed = model.forward(images)
            mse_rec_test += torchmetrics.MeanSquaredError()(reconstructed.cpu(), images)
            psnr_test += torchmetrics.image.PeakSignalNoiseRatio()(reconstructed.cpu(), images)
        mse_rec_test = mse_rec_test/len(testloader)
        psnr_test = psnr_test/(len(testloader))

        print('MSE (test) = ', mse_rec_test)
        print('PSNR (test) = ', psnr_test)


    # ---- Measure the reconstruction error (class error) ---- #
    with torch.no_grad():
        if dataset == 'FMNIST' or dataset=='MNIST':
            # Empty cache
            torch.cuda.empty_cache()

            # Load the classifier
            classifier_network = ClassifierNetwork().to(model.device)
            checkpoint = torch.load('classifier_network_'+str(dataset).lower()+'.pt', map_location=model.device)
            classifier_network.load_state_dict(checkpoint['model_state_dict'])

            # Evaluate the reconstruction error
            hit_score_test = eval_reconstruction(classifier_network, model, testloader)
            print('ACC IN RECONSTRUCTION (test) = ', hit_score_test)

            if bits_info<15:
                # Evaluate the reconstruction error with high probable words
                hit_score_test_conf = eval_reconstruction(classifier_network, model, testloader,code_words=all_info_words, threshold=0.4)
                print('CONF. ACC IN RECONSTRUCTION (test, threshold=0.4) = ', hit_score_test_conf)

if __name__ == '__main__':
    main()