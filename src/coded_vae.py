import torch
import torch.nn as nn
from torch import optim

from src.nn.encoder import Encoder
from src.nn.decoder import Decoder

from src.utils.sampling import sample_from_qz_given_x, sample_from_qc_given_x, modulate_words
from src.train.loss import compute_word_logprobs
from src.train.train import trainloop
from src.utils.functions import check_args, set_random_seed
from src.nn.modules import dclamp
from src.nn import modules


class CodedVAE(nn.Module):

    """
    Class implementing the Coded-DVAE.
    """

    def __init__(self, enc, dec, latent_dim, bits_info=None, G=None, A=None, likelihood='gauss', beta=10, lr=1e-4, weight_decay=1e-4, inference='word', seed=None, polar=False, device='cpu'):

        super(CodedVAE, self).__init__()
        """
        Initialize an instance of the class.

        Parameters
        ----------
        enc : torch.nn.Module
            Module with the architecture of the encoder neural network without the output activation.
        dec : torch.nn.Module
            Module with the architecture of the decoder neural network.
        latent_dim : int
            Latent dimension of the model.
        bits_info : list, optional
            List indicating the number of information bits in each branch. Sorted in ascending order (from m_1 to m_j).
        G : torch.tensor, optional
            Matrix used to encode information words.
        A: list of torch.tensor, optional
            List with the adjacency matrix of the different branches. Each matrix A idicate which information bits should 
            be combined. Sorted in ascenting order (A[0] indicates combination m_1+m_2, A[1] indicates combination m_1+m_2, and so on)
        likelihood: string, optional
            Distribution used to compute the reconstruction term. Default 'gauss'.
            - 'gauss': Gaussian likelihood.
            - 'ber': Bernoulli likelihood.
        beta: float, optional
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        lr: float, optional
            Learning rate. Default to 1e-4.
        weight_decay: float
            Weight decay. Default to 1e-4.
        inference: string
            Inference type. Default 'rep'.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
            - 'hier' for the coded hierarchical case.
        seed: int
            Seed for reproducibility.
        polar: boolean, optional
            Indicate if using the polar code or no. Default to False.
        """

        # Configuration
        self.beta = torch.tensor(beta)
        self.likelihood = likelihood
        self.latent_dim = latent_dim
        self.inference = inference

        # Code matrices
        self.G = G

        if self.inference == 'hier':
            if G != None:
                self.H = [torch.transpose(G_i, 0, 1) for G_i in G]
            else:
                self.H = None
        else:
            if G != None:
                # G is not a list of tensors but rather one tensor
                self.H = G.T
            else:
                self.H = None

        # Flag for using the polar code in the hierarchical case (only used in the hierarchical model)
        self.polar = polar
        
        # Adjacency matrices (only used in the hierarchical model)
        self.A = A
        
        # Bits info
        if bits_info is None:
            self.bits_info = self.latent_dim
        else:
            self.bits_info = bits_info

        # If the bits_info is not a list then we can just use it as the parameter
        if self.inference == 'hier':
            if type(self.bits_info) != int:
                self.bits_info_total = sum(self.bits_info)
            else:
                self.bits_info_total = self.bits_info
        else:
            self.bits_info_total = self.bits_info

        # Check arguments
        # It was assigning self.H as the G and G as the codewords because of the order it is in. SPECIFIED THE ORDER BY SAYING H=
        check_args(self.inference, G = self.G, H = self.H)

        # Encoder
        self.encoder = Encoder(enc, inference_type=self.inference)
        # Decoder
        self.decoder = Decoder(dec)

        # Optimizers
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer_decoder = optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=weight_decay)

        # Set random seed
        if not (seed is None):
            set_random_seed(seed)

        # Device
        self.device = device
        

    def forward(self, x):

        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """

        x = x.to(self.device)
        
        # Forward encoder
        encoder_out = self.encoder.forward(x)

        # Sanity check
        assert torch.any(torch.isinf(encoder_out))==False, "Invalid probs value (inf)."
        assert torch.any(torch.isnan(encoder_out))==False, "Invalid probs value (nan)."

        # Sample from the latent distribution
        # Uncoded case
        if self.inference == 'uncoded':
            z_sample = sample_from_qz_given_x(encoder_out, beta=self.beta, n_samples=1)

        # Coded case (repetition codes) 
        if self.inference == 'rep':
            
            # Soft decoding: obtain the marginals of the information bits
            logpm1 = torch.matmul(torch.log(encoder_out), self.H.to(self.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out), self.H.to(self.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            # Introduce code structure
            qc = torch.matmul(torch.exp(log_marginals_norm[:,:,1]), self.G.to(self.device))
            # Modulate c to obtain z
            z_sample = sample_from_qz_given_x(qc, beta=self.beta, n_samples=1)
        
        # Coded Hierarchical case
        if self.inference == 'hier':
            
            log_marginals_norm = torch.zeros(x.shape[0], self.bits_info_total, 2).to(self.device) # Marginals of the information bits
            log_qc_polar =  torch.zeros(x.shape[0], self.bits_info_total).to(self.device)  # Marginals taking into account the polar code (combination of the info bits), before the repetition code

            # Bit probabilities in the first branch are always computed directly:
            logpm1 = torch.matmul(torch.log(encoder_out[:,:self.H[0].shape[0]]), self.H[0].to(self.device))
            logpm0 = torch.matmul(torch.log(1-encoder_out[:,:self.H[0].shape[0]]), self.H[0].to(self.device))

            log_marginals = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_norm[:,:self.bits_info[0],:] = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

            log_qc_polar[:,:self.bits_info[0]] = log_marginals_norm[:,:self.bits_info[0],1]
            qc_rep = torch.matmul(torch.exp(log_qc_polar[:,:self.bits_info[0]]),self.G[0].to(self.device))
        
            cdim = self.H[0].shape[0]
            mdim = self.bits_info[0]

            # Iterate over the rest of the branches
            for i in range(1,len(self.bits_info)):
                
                # Soft decoding: obtain the marginals of the information bits for branch x
                logpmx1 = torch.matmul(torch.log(encoder_out[:,cdim:(cdim+self.H[i].shape[0])]), self.H[i].to(self.device))
                logpmx0 = torch.matmul(torch.log(1-encoder_out[:,cdim:(cdim+self.H[i].shape[0])]), self.H[i].to(self.device))

                log_marginals_mx = torch.stack((logpmx0, logpmx1), dim=2)
                log_marginals_mx_norm = log_marginals_mx - torch.logsumexp(log_marginals_mx, dim=-1, keepdim=True)

                # Leverage the structure of the polar code to recover the marginals of the information bits
                combination01 = torch.matmul(log_marginals_norm[:,(mdim-self.bits_info[i-1]):mdim,0], self.A[i-1].to(self.device))+log_marginals_mx_norm[:,:,1] # log q(m1=0|x) + log q((m1+m2)=1|x)) 
                combination10 = torch.matmul(log_marginals_norm[:,(mdim-self.bits_info[i-1]):mdim,1], self.A[i-1].to(self.device))+log_marginals_mx_norm[:,:,0] # log q(m1=1|x) + log q((m1+m2)=0|x)) 
                combination = torch.stack((combination01, combination10), dim=2) 

                # Marginals of the information bits
                logpm1 = torch.logsumexp(combination, dim=-1) 
                logpm0 = torch.log(1-torch.exp(logpm1))

                log_marginals_info = torch.stack((logpm0, logpm1), dim=2) 
                log_marginals_info_norm = log_marginals_info - torch.logsumexp(log_marginals_info, dim=-1, keepdim=True) #already normalized

                # Sanity check
                assert torch.any(torch.exp(log_marginals_norm)>1.)==False, "Invalid probs value (p>1)."
                assert torch.any(torch.exp(log_marginals_norm)<0.)==False, "Invalid probs value (p<0)."
                
                log_marginals_norm[:,mdim:(mdim+self.bits_info[i]),0] = log_marginals_info_norm[:,:,0]
                log_marginals_norm[:,mdim:(mdim+self.bits_info[i]),1] = log_marginals_info_norm[:,:,1]

                # Re-compute the marginals of the polar-coded bits
                logqmx1 = torch.logsumexp(torch.stack((torch.matmul(log_marginals_norm[:,(mdim-self.bits_info[i-1]):mdim,0], self.A[i-1].to(self.device))+log_marginals_info_norm[:,:,1], torch.matmul(log_marginals_norm[:,(mdim-self.bits_info[i-1]):mdim,1], self.A[i-1].to(self.device))+log_marginals_info_norm[:,:,0]), dim=2), dim=2) 
                logqmx0 = torch.logsumexp(torch.stack((torch.matmul(log_marginals_norm[:,(mdim-self.bits_info[i-1]):mdim,0], self.A[i-1].to(self.device))+log_marginals_info_norm[:,:,0], torch.matmul(log_marginals_norm[:,(mdim-self.bits_info[i-1]):mdim,1], self.A[i-1].to(self.device))+log_marginals_info_norm[:,:,1]), dim=2), dim=2) 

                log_marginals_mx_norm = torch.stack((logqmx0, logqmx1), dim=2)
                log_marginals_mx_norm = log_marginals_mx - torch.logsumexp(log_marginals_mx, dim=-1, keepdim=True)

                log_qc_polar[:,mdim:(mdim+self.bits_info[i])] = log_marginals_mx_norm[:,:,1]

                qc_rep = torch.cat((qc_rep, torch.matmul(torch.exp(log_qc_polar[:,mdim:(mdim+self.bits_info[i])]), self.G[i].to(self.device))), dim=1)

                cdim += self.H[i].shape[0]
                mdim += self.bits_info[i]
            
            z_sample = sample_from_qz_given_x(qc_rep, beta=self.beta, n_samples=1)  # shape [N, K, n_samples]
        

        # Forward decoder
        reconstructed = self.decoder.forward(z_sample[:,:,0])

        return z_sample, reconstructed
    
    
    def train(self, train_dataloader, n_epochs=100, n_epochs_wu=0, start_epoch=0, n_samples=1, train_enc=True, train_dec=True, verbose=True, wb=False):

        """
        Train the model for a given number of epochs.
            
            Parameters
            ----------
            model : CodedDVAE instance
                Model to be trained.
            train_dataloader : torch Dataloader
                Dataloader with the training set.
            n_epochs: int, optional
                Number of epochs. Default 100.
            n_epochs_wu: int
                Number of warmup epochs. For the first n_epochs_wu the model is trained in 'uncoded' mode.
            start_epoch: int, optional
                Epoch where the trainloop starts. This is useful to obtain coherent logs in weights and biases when we finetune a model.
            n_samples : int, optional
                Number of samples used for computing the ELBO. The number of samples is 1 by default.
            train_enc : boolean, optional
                Flag to indicate if the parameters of the encoder need to be updated. True by default.
            train_enc : boolean, optional
                Flag to indicate if the parameters of the decoder need to be updated. True by default.
            verbose: boolean, optional
                Flag to print the ELBO during training. True by default.
            wb: boolean, optional
                Flag to log the ELBO, KL term and reconstruction term to Weights&Biases.

            Returns
            -------
            elbo_evol : list
                List containing the ELBO values obtained during training (1 value per epoch).
            kl_div_evol : list
                List containing the Kullback-Leibler divergence values obtained during training (1 value per epoch).
            reconstruction_evol : list
                List containing reconstruction term values obtained during training (1 value per epoch).
        """

        # Track loss evolution during training
        elbo_evol = []
        kl_evol = []
        rec_evol = []    

        if n_epochs_wu>0:
            # Warmup!
            print('Starting warmup...')
            elbo_evol_wu, kl_evol_wu, rec_evol_wu = trainloop(
                self, 
                train_dataloader, 
                n_epochs_wu, 
                start_epoch=start_epoch, 
                n_samples=n_samples, 
                train_enc=train_enc,
                train_dec=train_dec,
                inference='uncoded',
                verbose=verbose,
                wb=wb)
            
            elbo_evol.append(elbo_evol_wu)
            kl_evol.append(kl_evol_wu)
            rec_evol.append(rec_evol_wu)
            print('Warmup finished!')

        # Train!
        print('Starting training...')
        elbo_evol_train, kl_evol_train, rec_evol_train = trainloop(
            self, 
            train_dataloader, 
            n_epochs+n_epochs_wu, 
            start_epoch=n_epochs_wu, 
            n_samples=n_samples, 
            train_enc=train_enc,
            train_dec=train_dec,
            inference=self.inference,
            verbose=verbose,
            wb=wb 
        )

        elbo_evol.append(elbo_evol_train)
        kl_evol.append(kl_evol_train)
        rec_evol.append(rec_evol_train)
        print('Training finished!')

        return elbo_evol, kl_evol, rec_evol
    

    def generate(self, n_samples=100, m=None):

        """
        Generate new samples following the generative model.

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate.
        m: list, optional
            List with fixed information vectors. Assume teh first position corresponds to m1, second to m2, and so on.

        Returns
        -------
        Generated samples.

        """
        
        # Uncoded case
        if self.inference=='uncoded':
            # Uniform distribution for each bit
            m_probs = torch.ones((n_samples, self.latent_dim))*0.5
            # Sample z
            z_sample = sample_from_qz_given_x(m_probs.to(self.device), beta=self.beta, n_samples=1)
            # Forward decoder
            reconstructed = self.decoder.forward(z_sample[:,:,0])

        # Coded case (repetition codes)
        if self.inference=='rep':
            # Uniform distribution for each bit
            m_probs = torch.ones((n_samples, self.bits_info))*0.5
            # Sample information codewords
            m_sample = m_probs.bernoulli()
            
            # Obtain a codeword
            c = torch.matmul(m_sample, self.G)
            # Sample z
            z_sample = modulate_words(c.to(self.device), beta=self.beta)
            # Forward decoder
            reconstructed = self.decoder.forward(z_sample[:,:])

        # Coded Hierarchical case
        if self.inference=='hier':

            # Uniform distribution for each bit
            probs = torch.ones((n_samples, self.bits_info_total))*0.5
            m_sample = probs.bernoulli()

            # Fix bit values if indicated
            if m is not None:
                mdim=0
                for i in range(len(m)):
                    m_sample[:,mdim:(mdim+self.bits_info[i])] = torch.ones((n_samples, self.bits_info[i]))*m[i]
                    mdim += self.bits_info[i]
            
            # Obtain codewords
            c = torch.matmul(m_sample[:,:self.bits_info[0]], self.G[0])
            mdim=self.bits_info[0]
            for i in range(1, len(self.bits_info)):
                m_mixed = torch.fmod(torch.matmul(m_sample[:,(mdim-self.bits_info[i-1]):mdim], self.A[i-1]) + m_sample[:,mdim:(mdim+self.bits_info[i])], 2)
                c_i = torch.matmul(m_mixed, self.G[i])
                c = torch.cat((c,c_i), dim=1)
                mdim += self.bits_info[i]

            # Obtain z
            z_sample = modulate_words(c.to(self.device), beta=self.beta)

            # Forward decoder
            reconstructed = self.decoder.forward(z_sample[:,:])

        return reconstructed
    

    def save(self, path):

        """
        Save model.

        Parameters
        ----------
        path: str
           Path where the model will be saved.

        """

        torch.save(self.state_dict(), path)
        print('Model saved at ' + path)





