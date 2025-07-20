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

    # ---- Obtain Codebook ---- #
    if bits_info[0]<15:
        # Generate all possible words with bits_info
        all_info_words_1 = torch.FloatTensor(list(map(list, itertools.product([0, 1], repeat=bits_info[0]))))
        n_words_1 = all_info_words_1.shape[0]
        # Encode the info words
        all_words_1 = torch.matmul(all_info_words_1, rep_matrices[0]['G'])

    # ---- Load data ---- #
    os.makedirs("data/", exist_ok=True)

    dataset = config['eval_config']['dataset'] # Dataset can be 'MNIST', 'FMNIST', 'CIFAR10', or 'IMAGENET' 
    batch_size = config["eval_config"]["batch_size"]
    data_label = {}

    # MNIST # 
    if dataset == 'MNIST':

        D = 28*28
        # Download and load the test data
        testset = datasets.MNIST('../data/', download=False, train=False, transform=transforms.Compose([transforms.ToTensor()]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        # Generate a dictionary with the images organized by label (key:label, value: images)
        n_labels = 10
        for images, labels in testloader:
            for image, label in zip(images, labels):
                if label.item() in data_label:
                    data_label[label.item()] = torch.cat((data_label[label.item()], image.unsqueeze(0)), dim=0)  
                else:
                    data_label[label.item()] = image.unsqueeze(0)
        
    # FMNIST #
    if dataset == 'FMNIST':

        D = 28*28
        # Download and load the test data
        testset = datasets.FashionMNIST('./data/FMNIST/', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, generator=g)

        # Generate a dictionary with the images organized by label (key:label, value: images)
        n_labels = 10
        for images, labels in testloader:
            for image, label in zip(images, labels):
                if label.item() in data_label:
                    data_label[label.item()] = torch.cat((data_label[label.item()], image.unsqueeze(0)), dim=0) 
                else:
                    data_label[label.item()] = image.unsqueeze(0)

    if dataset == 'CIFAR10':

        D=32*32*3
        # Download and load the test data
        testset = datasets.CIFAR10('./data/CIFAR10/', download=True, train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        # Generate a dictionary with the images organized by label (key:label, value: images)
        n_labels = 10
        for images, labels in testloader:
            for image, label in zip(images, labels):
                if label.item() in data_label:
                    data_label[label.item()] = torch.cat((data_label[label.item()], image.unsqueeze(0)), dim=0) 
                else:
                    data_label[label.item()] = image.unsqueeze(0)


    if dataset == 'IMAGENET':

        D=64*64*3
        # Download and load the test data
        testset = datasets.ImageFolder('./data/tiny-imagenet-200/test', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


    # ---- Get encoder and decoder networks---- #
    enc = mod.get_encoder(config['model_config']['type_encoder'], bits_code_total, dataset)
    dec = mod.get_decoder(config['model_config']['type_decoder'], bits_code_total, dataset)

    # ---- Declare model ---- #
    # ---- Declare model ---- #
    model = CodedVAE(enc, dec, bits_code_total, bits_info=bits_info, likelihood=likelihood, G=G_list, A=A, beta=config['model_config']['beta'], lr=config['train_config']['lr'], inference=inf_type, seed=0, device=device, polar=polar)
    model.to(device)

    # ---- Load pre-trained model ---- #
    checkpoint = torch.load('checkpoints/'+config['eval_config']['checkpoint'], map_location=model.device) # Change the path with your own checkpoint!
    model.load_state_dict(checkpoint)
    print('Model loaded!')

    # Evaluation mode
    model.encoder.eval()
    model.decoder.eval()

    with torch.no_grad():
        n=50
        ber = torch.zeros(n)
        ber_map = torch.zeros(n)
        wer = torch.zeros(n)
        wer_map = torch.zeros(n)
        ber_branch = torch.zeros(n, len(bits_info))
        ber_branch_map = torch.zeros(n, len(bits_info))
        wer_branch = torch.zeros(n, len(bits_info))
        wer_branch_map = torch.zeros(n, len(bits_info))
        ber_per_bit = torch.zeros((n, bits_info_total))
        ber_per_bit_map = torch.zeros((n, bits_info_total))

        for j in range(n):
            
            n_word_samples = 1000

            # 1. Sample random words (uniform distribution for each bit)    
            probs = torch.ones((n_word_samples, bits_info_total))*0.5
            m_sample = probs.bernoulli()
            # Obtain codewords
            c = torch.matmul(m_sample[:,:bits_info[0]], rep_matrices[0]['G'])
            mdim=bits_info[0]
            for i in range(1, len(bits_info)):
                m_mixed = torch.fmod(torch.matmul(m_sample[:,(mdim-bits_info[i-1]):mdim], A[i-1]) + m_sample[:,mdim:(mdim+bits_info[i])], 2)
                c_i = torch.matmul(m_mixed, rep_matrices[i]['G'])
                c = torch.cat((c,c_i), dim=1)
                mdim += bits_info[i]

            z_sample = modulate_words(c, beta=torch.tensor(config['model_config']['beta']))

            # 2. Forward the sampled words through the decoder to obtain reconstructed images
            decoder_out =  model.decoder.forward(z_sample.to(model.device))

            # 3. Forward the reconstructed image through the encoder to obtain the recovered word
            encoder_out = model.encoder.forward(decoder_out)

            log_marginals_norm = torch.zeros(n_word_samples, bits_info_total, 2).to(model.device) # Marginals of the information bits
            log_qc_polar =  torch.zeros(n_word_samples, bits_info_total).to(model.device)  # Marginals taking into account the polar code (combination of the info bits), before the repetition code

            # Bit probabilities in the first branch are always computed directly
            logpm1 = torch.matmul(torch.log(encoder_out[:,:bits_code[0]]), rep_matrices[0]['D'].to(model.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out[:,:bits_code[0]]), rep_matrices[0]['D'].to(model.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm[:,:bits_info[0],:] = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            log_qc_polar[:,:bits_info[0]] = log_marginals_norm[:,:bits_info[0],1]
            qc_rep = torch.matmul(torch.exp(log_qc_polar[:,:bits_info[0]]),rep_matrices[0]['G'].to(model.device))
        
            cdim = bits_code[0]
            mdim = bits_info[0]
            for i in range(1,len(bits_info)):
                
                # Compute mixed marginals
                logpmx1 = torch.matmul(torch.log(encoder_out[:,cdim:(cdim+bits_code[i])]), rep_matrices[i]['D'].to(model.device))
                logpmx0 = torch.matmul(torch.log(1-encoder_out[:,cdim:(cdim+bits_code[i])]), rep_matrices[i]['D'].to(model.device))

                log_marginals_mx = torch.stack((logpmx0, logpmx1), dim=2)
                log_marginals_mx_norm = log_marginals_mx - torch.logsumexp(log_marginals_mx, dim=-1, keepdim=True)

                # Leverage the structure of the polar code to recover the marginals of the information bits
                combination01 = torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,0], A[i-1].to(model.device))+log_marginals_mx_norm[:,:,1] # log q(m1=0|x) + log q((m1+m2)=1|x)) 
                combination10 = torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,1], A[i-1].to(model.device))+log_marginals_mx_norm[:,:,0] # log q(m1=1|x) + log q((m1+m2)=0|x)) 
                combination = torch.stack((combination01, combination10), dim=2)

                # Marginals of the information bits
                logpm1 = torch.logsumexp(combination, dim=-1)
                logpm0 = torch.log(1-torch.clamp(torch.exp(logpm1), 0.0001, 0.9999)) # avoid numerical instabilities

                log_marginals_info = torch.stack((logpm0, logpm1), dim=2) 
                log_marginals_info_norm = log_marginals_info - torch.logsumexp(log_marginals_info, dim=-1, keepdim=True) #already normalized

                # Sanity check
                assert torch.any(torch.exp(log_marginals_norm)>1.)==False, "Invalid probs value (p>1)."
                assert torch.any(torch.exp(log_marginals_norm)<0.)==False, "Invalid probs value (p<0)."
                
                log_marginals_norm[:,mdim:(mdim+bits_info[i]),0] = log_marginals_info_norm[:,:,0]
                log_marginals_norm[:,mdim:(mdim+bits_info[i]),1] = log_marginals_info_norm[:,:,1]

                # Re-compute the marginals of the polar-coded bits
                logqmx1 = torch.logsumexp(torch.stack((torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,0], A[i-1].to(model.device))+log_marginals_info_norm[:,:,1], torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,1], A[i-1].to(model.device))+log_marginals_info_norm[:,:,0]), dim=2), dim=2) 
                logqmx0 = torch.logsumexp(torch.stack((torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,0], A[i-1].to(model.device))+log_marginals_info_norm[:,:,0], torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,1], A[i-1].to(model.device))+log_marginals_info_norm[:,:,1]), dim=2), dim=2) 

                log_marginals_mx_norm = torch.stack((logqmx0, logqmx1), dim=2)
                log_marginals_mx_norm = log_marginals_mx - torch.logsumexp(log_marginals_mx, dim=-1, keepdim=True)

                log_qc_polar[:,mdim:(mdim+bits_info[i])] = log_marginals_mx_norm[:,:,1]
                qc_rep = torch.cat((qc_rep, torch.matmul(torch.exp(log_qc_polar[:,mdim:(mdim+bits_info[i])]), rep_matrices[i]['G'].to(model.device))), dim=1)

                cdim += bits_code[i]
                mdim += bits_info[i]

            # General BER and WER
            word_recovered = torch.bernoulli(torch.exp(log_marginals_norm[:,:,1]))
            word_recovered_map = torch.where(torch.exp(log_marginals_norm[:,:,1])> 0.5 , 1., 0.)
            # Sampled
            ber[j] = (m_sample != word_recovered.cpu().data).sum()/(m_sample.shape[0]*bits_info_total) # Bit Error Rate = n_bit_errors/n_bits_transmitted
            ber_per_bit[j,:] = (m_sample != word_recovered.cpu().data).sum(dim=0)/m_sample.shape[0]
            wer[j] = ((m_sample != word_recovered.cpu().data).sum(dim=1) != 0).sum()/m_sample.shape[0]   # Word Error Rate = n_words_incorrect/n_words_transmitted
            # MAP
            ber_map[j] = (m_sample != word_recovered_map.cpu().data).sum()/(m_sample.shape[0]*bits_info_total) # Bit Error Rate = n_bit_errors/n_bits_transmitted
            ber_per_bit_map[j,:] = (m_sample != word_recovered_map.cpu().data).sum(dim=0)/m_sample.shape[0]
            wer_map[j] = ((m_sample != word_recovered_map.cpu().data).sum(dim=1) != 0).sum()/m_sample.shape[0]   # Word Error Rate = n_words_incorrect/n_words_transmitted

            # BER and WER per branch
            mdim = 0
            for i in range(len(bits_info)):
                m_i = m_sample[:,mdim:(mdim+bits_info[i])]
                word_recovered_i = word_recovered.cpu().data[:,mdim:(mdim+bits_info[i])]
                word_recovered_map_i=  word_recovered_map.cpu().data[:,mdim:(mdim+bits_info[i])]
                # Sampled
                ber_branch[j,i] = (m_i != word_recovered_i).sum()/(m_i.shape[0]*bits_info[i]) # Bit Error Rate = n_bit_errors/n_bits_transmitted
                wer_branch[j,i] = ((m_i != word_recovered_i).sum(dim=1) != 0).sum()/m_i.shape[0]   # Word Error Rate = n_words_incorrect/n_words_transmitted
                #MAP
                ber_branch_map[j,i] = (m_i != word_recovered_map_i).sum()/(m_i.shape[0]*bits_info[i]) # Bit Error Rate = n_bit_errors/n_bits_transmitted
                wer_branch_map[j,i] = ((m_i != word_recovered_map_i).sum(dim=1) != 0).sum()/m_i.shape[0]   # Word Error Rate = n_words_incorrect/n_words_transmitted
                mdim += bits_info[i]

        print('BER (sampled): ', torch.mean(ber).item())
        print('BER (MAP): ', torch.mean(ber_map).item())
        print('WER (sampled): ', torch.mean(wer).item())
        print('WER (MAP): ', torch.mean(wer_map).item())

    # ---- Generation ---- #
    with torch.no_grad():

        # 1. Generate random images
        generated = model.generate(n_samples=100)
        if dataset=='CIFAR10' or dataset=='SVHN' or dataset=='IMAGENET':
            generated=generated*0.5 + 0.5
        # 2. Plot generated images
        fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(20,20))
        if dataset=='CIFAR10'or dataset=='SVHN' or dataset=='IMAGENET':
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


        all_info_words_2 = torch.FloatTensor(list(map(list, itertools.product([0, 1], repeat=bits_info[1]))))
        # ---- Generation m1 fixed ---- #
        if bits_info[0] < 15:
            with torch.no_grad():
                for w in range(all_info_words_1.shape[0]):

                    # 1. Generate random images
                    generated = model.generate(n_samples=100, m=[all_info_words_1[w]])
                    if dataset=='CIFAR10' or dataset=='SVHN' or dataset=='IMAGENET' or dataset=='CELEBA':
                        generated=generated*0.5 + 0.5
                    # 2. Plot generated images
                    fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(20,20))
                    if dataset=='CIFAR10'or dataset=='SVHN' or dataset=='IMAGENET' or dataset=='CELEBA':
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
                    # Add a main title for the entire figure
                    #fig.suptitle('m1 = '+str(all_info_words_1[w].numpy())+' m2 = '+str(all_info_words_2[3].numpy()), fontsize=30)
                    fig.suptitle('m1 = '+str(all_info_words_1[w].numpy()), fontsize=30)

                    # Remove padding between subplots
                    # plt.subplots_adjust(wspace=0, hspace=0)
                    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0, hspace=0)
                    plt.show()

    # ---- Reconstruction ---- #
    with torch.no_grad():

        # 1. Obtain a batch of test data
        test = iter(testloader)
        images, labels = next(test)
        # 2. Forward model
        _, reconstructed = model.forward(images)
        # 3. Plot reconstructed images
        if dataset=='CIFAR10' or dataset=='SVHN' or dataset=='IMAGENET' or dataset=='CELEBA':
            images=images*0.5 + 0.5
            reconstructed=reconstructed*0.5 + 0.5
        fig, axes = plt.subplots(nrows=2, ncols=20, sharex=True, sharey=True, figsize=(40,4))
        for i in range(20):
            if dataset=='CIFAR10' or dataset=='SVHN' or dataset=='IMAGENET' or dataset=='CELEBA':
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

        print('MSE reconstruction test: ', mse_rec_test.item())
        print('PSNR reconstruction test: ', psnr_test.item())

if __name__ == '__main__':
    main()