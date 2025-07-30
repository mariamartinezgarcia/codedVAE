import torch
import torch.distributions as dist
from src.utils.sampling import sample_from_qz_given_x, sample_from_qc_given_x, modulate_words
from src.nn.modules import dclamp

def log_bernoulli(probs, observation):

    """
    Evaluate a Bernoulli distribution.
        
        Parameters
        ----------
        probs : torch.tensor
            Tensor with the probabilities to define the Bernoulli. [shape (batch_size, dimension)]
        observation: torch.tensor
            Batch of data.
    
        Returns
        -------
        Log probability.
    """

    bce = torch.nn.BCELoss(reduction='none')

    return -torch.sum(bce(probs, observation), dim=1)


def log_gaussian(x, mean, covar):

    """
    Evaluate a Multivariate Gaussian distribution with diagonal covariance matrix.
        
        Parameters
        ----------
        x : torch.tensor
            Batch of data.
        mean : torch.tensor
            Means of the distribution.
        covar : torch.tensor
            Value of the diagonal.

        Returns
        -------
        Log probability.
    """

    # MVN INDEPENDEN NORMAL DISTRIBUTIONS
    # Create a multivariate normal distribution with diagonal covariance
    gaussian = dist.independent.Independent(dist.Normal(mean, torch.sqrt(covar)), 1)
    
    return gaussian.log_prob(x)


def kl_div_bernoulli(q_probs, p_probs):

    """
    Compute KL Divergence D_KL(q|p) between two Bernoulli distributions.

    Parameters
        ----------
        q_probs : torch.tensor
            Probabilities that define the q distribution.
        p_probs : torch.tensor
           Probabilities that define the p distribution.

        Returns
        -------
        kl_div : torch.tensor
            Kullback-Leibler divergence between the given distributions.
    """

    q = dist.Bernoulli(dclamp(q_probs, min=0, max=1-1e-3)) # clamp to avoid numerical instabilities
    p = dist.Bernoulli(p_probs)

    kl_div = dist.kl.kl_divergence(q, p)

    kl_div = torch.sum(kl_div, dim=1)

    return kl_div


def compute_word_logprobs(bit_probs, code_words):

    """
    Compute the log probability of the words in the codebook.

    Parameters
        ----------
        bit_probs : torch.tensor
            Bit probabilities.
        code_words : torch.tensor
            Matrix containing the codebook.

        Returns
        -------
        logq : torch.tensor
            Unnormalized distribution over words.
        logq_norm : torch.tensor
            Normalized distribution over words.
    """

    # Sanity check
    assert torch.any(bit_probs < 0)==False, "Negative value encountered in bit probabilities."
    assert torch.any(bit_probs > 1)==False, "Value larger than 1 encountered in bit probabilities."
    assert torch.all(torch.logical_or(code_words == 0, code_words == 1)), "Invalid word encountered. All words should be binary vectors."

    # === Compute log(q(c|x,C)) [evaluate log(q_uncoded(c|x)) for code words] === #

    # 1. Extend the output of the encoder in a third dimension to obtain a tensor of shape [batch_size, K, n_words]
    # 2. Extend the code words matrix in a third dimension to obtain a tensor of shape [batch_size, K, n_words]
    # 3. Reduce the logq in dim=1 to obtain a matrix of shape [batch_size, n_words] containing the evaluation of log(q(c|x,C)) for each code word
    
    n_words = code_words.shape[0] 
    batch_size = bit_probs.shape[0]

    # Clamp to avoid numerical instabilities
    bit_probs = dclamp(bit_probs, min=0.001, max=0.999)

    # Evaluate log(q_uncoded(c|x)) for code words
    logq = log_bernoulli(bit_probs.unsqueeze(2).repeat(1, 1, n_words), code_words.T.unsqueeze(0).repeat(batch_size,1,1))
    
    # Clamp to avoid numerical instabilities
    logq = dclamp(logq, min=-100, max=1)

    # Sanity check
    assert torch.any(torch.isinf(logq))==False, "Invalid logq value (inf)."
    assert torch.any(torch.isnan(logq))==False, "Invalid logq value (nan)."

    # Normalization
    logq_norm = logq - logq.logsumexp(dim=-1, keepdim=True)

    # Sanity check
    assert torch.all(torch.exp(logq_norm) >= 0), "Negative value encountered in normalized probs."
    assert torch.all((torch.exp(logq_norm).sum(-1) - 1).abs() < 1e-5), "Normalized probabilities do not sum 1."

    return logq, logq_norm


def get_elbo_uncoded(x, encoder, decoder, prior_m=0.5, beta=10, likelihood='gauss', n_samples=1):

    """
    Compute the ELBO in the uncoded scenario.

    Parameters
        ----------
        x : torch.tensor
            Batch of data.
        encoder : Encoder instance
            Encoder of the model.
        decoder : Decoder instance
            Decoder of the model.
        prior_m : float, optional
            Prior bit probability. Default to 0.5.
        beta: float, optional
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        likelihood: string, optional
            Distribution used to compute the reconstruction term.
            - 'gauss': Gaussian likelihood.
            - 'ber': Bernoulli likelihood.
        n_samples: int, optional
            Number of samples used to estimate the ELBO. Default to 1.

        Returns
        -------
        elbo : torch.tensor
            Value of the ELBO.
        kl_div : torch.tensor
            Value of the Kullback-Leibler divergence term in the ELBO.
        reconstruction: torch.tensor
            Value of the reconstruction term in the ELBO.
    """

    N = x.shape[0]
    x_flat = x.view(N,-1)

    # Forward encoder
    bit_probs = encoder.forward(x)

    # Obtain n_samples from q(z|x) for each observed x
    qz_sample = sample_from_qz_given_x(bit_probs, beta=beta, n_samples=n_samples)  # shape [N, K, n_samples]

    # Compute the reconstruction term E_{q(z|x)}[log p(x|z)]
    reconstruction_sum = 0

    for n in range(n_samples):
        
        # Forward decoder
        out_decoder = decoder.forward(qz_sample[:, :, n]).view(-1, x_flat.shape[1])

        # Binary observation model
        if likelihood.lower() == 'ber':
            reconstruction_sum += log_bernoulli(out_decoder, x_flat)
        # Real observation model
        elif likelihood.lower() == 'gauss':
            covar = torch.ones(out_decoder.shape[1]).to(x_flat.device) * 0.1
            reconstruction_sum += log_gaussian(x_flat, out_decoder, covar)    # Fixed variance

    reconstruction = reconstruction_sum/n_samples

    # Compute the KL Divergence term
    prior_probs = (torch.ones(bit_probs.shape)*prior_m).to(x_flat.device)
    kl_div = kl_div_bernoulli(bit_probs, prior_probs)

    # Obtain the ELBO loss
    elbo = torch.sum((reconstruction-kl_div), dim=0)/N

    return elbo, torch.sum(kl_div, dim=0)/N, torch.sum(reconstruction, dim=0)/N

def get_iwae_uncoded(x, encoder, decoder, prior_m=0.5, beta=10, likelihood='gauss', n_samples=50):

    N = x.shape[0]
    x_flat = x.view(N,-1)
    
    # Forward encoder
    bit_probs = encoder.forward(x)

    # Obtain n_samples from q(z|x) for each observed x
    qz_sample = sample_from_qz_given_x(bit_probs, beta=beta, n_samples=n_samples)  # shape [N, K, n_samples]

    # Obtain prior z
    logpz_m0 = torch.log(torch.tensor(0.5)) + (-beta*qz_sample) - torch.log((1-torch.exp(-beta))/beta)
    logpz_m1 = torch.log(torch.tensor(0.5)) + (beta*(qz_sample-1)) - torch.log((1-torch.exp(-beta))/beta)
    logpz = torch.stack((logpz_m0, logpz_m1), dim=-1)
    logpz = torch.logsumexp(logpz, dim=-1)
    logpz = torch.sum(logpz, dim=1) # Across dimensions (independent)

    # Obtain posterior z
    logqz_m0 = torch.log(1-(bit_probs.unsqueeze(2).repeat(1, 1, n_samples))) + (-beta*qz_sample) - torch.log((1-torch.exp(-beta))/beta)
    logqz_m1 = torch.log(bit_probs.unsqueeze(2).repeat(1, 1, n_samples)) + (beta*(qz_sample-1)) - torch.log((1-torch.exp(-beta))/beta)
    logqz = torch.stack((logqz_m0, logqz_m1), dim=-1)
    logqz = torch.logsumexp((logqz), dim=-1)
    logqz = torch.sum(logqz, dim=1)
    
    # Compute the reconstruction term E_{q(z|x)}[log p(x|z)]
    reconstruction = torch.zeros((N, n_samples)).to(x.device)
    for n in range(n_samples):
        
        # Forward decoder
        out_decoder = decoder.forward(qz_sample[:, :, n]).view(-1, x_flat.shape[1])

        # Binary observation model
        if likelihood.lower() == 'ber':
            reconstruction[:,n] = log_bernoulli(out_decoder, x_flat)
        # Real observation model
        elif likelihood.lower() == 'gauss':
            covar = torch.ones(out_decoder.shape[1]).to(x_flat.device) * 0.1
            reconstruction[:,n] = log_gaussian(x_flat, out_decoder, covar)    # Fixed variance

    logw = reconstruction + logpz - logqz
    logw_norm = logw - torch.logsumexp(logw, dim=1, keepdim=True) 

    w = logw_norm.exp().detach()
    iwae = (w * logw).sum(1).mean()

    kl_div = torch.mean(torch.sum((logqz-logpz), dim=1))

    return iwae, kl_div, torch.mean(torch.mean(reconstruction, dim=1), dim=0)


def get_elbo_rep(x, encoder, decoder, G, H, prior_m=0.5, beta=10, likelihood='gauss', n_samples=1):

    """
    Compute the ELBO in the uncoded scenario.

    Parameters
        ----------
        x : torch.tensor
            Batch of data.
        encoder : Encoder instance
            Encoder of the model.
        decoder : Decoder instance
            Decoder of the model.
        G : torch.tensor
            Matrix used to encode information words.
        H : torch.tensor
            Matrix used to decode coded words
        beta: float, optional
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        likelihood: string, optional
            Distribution used to compute the reconstruction term.
            - 'gauss': Gaussian likelihood.
            - 'ber': Bernoulli likelihood.
        n_samples: int, optional
            Number of samples used to estimate the ELBO. Default to 1.

        Returns
        -------
        elbo : torch.tensor
            Value of the ELBO.
        kl_div : torch.tensor
            Value of the Kullback-Leibler divergence term in the ELBO.
        reconstruction: torch.tensor
            Value of the reconstruction term in the ELBO.
    """

    N = x.shape[0]
    x_flat = x.view(N,-1)

    probs = encoder.forward(x)

    # Sanity check
    assert torch.any(torch.isinf(probs))==False, "Invalid probs value (inf)."
    assert torch.any(torch.isnan(probs))==False, "Invalid probs value (nan)."

    logpm1 = torch.matmul(torch.log(probs), H.to(probs.device))
    logpm0 = torch.matmul(torch.log(1-probs), H.to(probs.device))

    log_marginals = torch.stack((logpm0, logpm1), dim=2)

    log_marginals_norm = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

    # Introduce code structure
    qc = torch.matmul(torch.exp(log_marginals_norm[:,:,1]), G.to(probs.device))

    # Obtain n_samples from q(z|x) for each observed x
    qz_sample = sample_from_qz_given_x(qc, beta=beta, n_samples=n_samples)  # shape [N, K, n_samples]
    
    # Compute the reconstruction term E_{q(z|x)}[log p(x|z)]
    reconstruction_sum = 0

    for n in range(n_samples):

        # Forward decoder
        out_decoder = decoder.forward(qz_sample[:,:,n]).view(-1, x_flat.shape[1])

        assert torch.any(torch.isinf(out_decoder))==False, "Invalid out_decoder value (inf)."
        assert torch.any(torch.isnan(out_decoder))==False, "Invalid out_decoder value (nan)."

        # Binary observation model
        if likelihood.lower() == 'ber':
            reconstruction_sum += log_bernoulli(out_decoder, x_flat)
        # Real observation model
        elif likelihood.lower() == 'gauss':
            covar = torch.ones(out_decoder.shape[1]).to(x_flat.device) * 0.1
            reconstruction_sum += log_gaussian(x_flat, out_decoder, covar)    # Fixed variance

    reconstruction = reconstruction_sum/n_samples

    # Compute the KL Divergence term
    prior_probs = (torch.ones(logpm1.shape)*prior_m).to(x_flat.device)
    kl_div = kl_div_bernoulli(torch.exp(log_marginals_norm[:,:,1]), prior_probs)

    # Obtain the ELBO loss
    elbo = torch.sum((reconstruction-kl_div), dim=0)/N

    return elbo, torch.sum(kl_div, dim=0)/N, torch.sum(reconstruction, dim=0)/N


def get_elbo_hier(x, encoder, decoder, G, H, bits_info, bits_info_total, A, prior_m=0.5, beta=10, likelihood='gauss', n_samples=1, polar=True):

    """
    Compute the ELBO in the uncoded scenario.

    Parameters
        ----------
        x : torch.tensor
            Batch of data.
        encoder : Encoder instance
            Encoder of the model.
        decoder : Decoder instance
            Decoder of the model.
        G : torch.tensor
            Matrix used to encode information words.
        H : torch.tensor
            Matrix used to decode coded words
        bits_info: list
            List indicating the number of information bits in each branch. Sorted in ascending order (from m_1 to m_j).
        bits_info_total: int
            Total number of information bits (considering all branches).
        A: list of torch.tensor
            List with the adjacency matrix of the different branches. Each matrix A idicate which information bits should 
            be combined. Sorted in ascenting order (A[0] indicates combination m_1+m_2, A[1] indicates combination m_1+m_2, and so on)
        beta: float, optional
            Temperature term that controls the decay of the exponentials in the smoothing transformation. Default to 10.
        likelihood: string, optional
            Distribution used to compute the reconstruction term.
            - 'gauss': Gaussian likelihood.
            - 'ber': Bernoulli likelihood.
        n_samples: int, optional
            Number of samples used to estimate the ELBO. Default to 1.
        polar: boolean, optional
            Flag indicating if we should use the polar code in inference.

        Returns
        -------
        elbo : torch.tensor
            Value of the ELBO.
        kl_div : torch.tensor
            Value of the Kullback-Leibler divergence term in the ELBO.
        reconstruction: torch.tensor
            Value of the reconstruction term in the ELBO.
    """

    N = x.shape[0]
    x_flat = x.view(N,-1)

    G = [G_i.to(x.device) for G_i in G]
    H = [H_i.to(x.device) for H_i in H]
    A = [A_i.to(x.device) for A_i in A]

    kl_total = 0
    kl_list = []

    # Forward encoder -> q_u(c1|x), q_u(c2|x)
    probs = encoder.forward(x)

    # Sanity check
    assert torch.any(torch.isinf(probs))==False, "Invalid probs value (inf)."
    assert torch.any(torch.isnan(probs))==False, "Invalid probs value (nan)."

    log_marginals_norm = torch.zeros(x.shape[0], bits_info_total, 2).to(x_flat.device) # Marginals of the information bits
    log_qc_polar =  torch.zeros(x.shape[0], bits_info_total).to(x_flat.device)  # Marginals taking into account the polar code (combination of the info bits), before the repetition code

    # Bit probabilities in the first branch are always computed directly
    logpm1 = torch.matmul(torch.log(probs[:,:H[0].shape[0]]), H[0])
    logpm0 = torch.matmul(torch.log(1-probs[:,:H[0].shape[0]]), H[0])

    log_marginals = torch.stack((logpm0, logpm1), dim=2)
    log_marginals_norm[:,:bits_info[0],:] = log_marginals - torch.logsumexp(log_marginals, dim=-1, keepdim=True)

    kl_div = kl_div_bernoulli(torch.exp(log_marginals_norm[:,:bits_info[0],1]), (torch.ones(logpm1.shape)*prior_m).to(x_flat.device))
    kl_total += kl_div
    kl_list.append(torch.sum(kl_div, dim=0)/N)

    log_qc_polar[:,:bits_info[0]] = log_marginals_norm[:,:bits_info[0],1]
    qc_rep = torch.matmul(torch.exp(log_qc_polar[:,:bits_info[0]]),G[0])
 
    cdim = H[0].shape[0]
    mdim = bits_info[0]
    for i in range(1,len(bits_info)):

        if polar:
            # Compute mixed marginals
            logpmx1 = torch.matmul(torch.log(probs[:,cdim:(cdim+H[i].shape[0])]), H[i])
            logpmx0 = torch.matmul(torch.log(1-probs[:,cdim:(cdim+H[i].shape[0])]), H[i])

            log_marginals_mx = torch.stack((logpmx0, logpmx1), dim=2)
            log_marginals_mx_norm = log_marginals_mx - torch.logsumexp(log_marginals_mx, dim=-1, keepdim=True)

            # Leverage the structure of the polar code to recover the marginals of the information bits
            combination01 = torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,0], A[i-1])+log_marginals_mx_norm[:,:,1] # log q(m1=0|x) + log q((m1+m2)=1|x)) 
            combination10 = torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,1], A[i-1])+log_marginals_mx_norm[:,:,0] # log q(m1=1|x) + log q((m1+m2)=0|x)) 
            combination = torch.stack((combination01, combination10), dim=2)

            # Marginals of the information bits
            logpm1 = torch.logsumexp(combination, dim=-1)
            logpm0 = torch.log(1-dclamp(torch.exp(logpm1), 0.0001, 0.9999)) # avoid numerical instabilities

            log_marginals_info = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_info_norm = log_marginals_info - torch.logsumexp(log_marginals_info, dim=-1, keepdim=True)
            # Sanity check
            assert torch.any(torch.sum(torch.exp(log_marginals_info_norm), dim=-1)>(1+1e-4))==False, "Invalid probs value (sum>1)."
            assert torch.any(torch.sum(torch.exp(log_marginals_info_norm), dim=-1)<(1-1e-4))==False, "Invalid probs value (sum<1)."

            log_marginals_norm[:,mdim:(mdim+bits_info[i]),0] = log_marginals_info_norm[:,:,0]
            log_marginals_norm[:,mdim:(mdim+bits_info[i]),1] = log_marginals_info_norm[:,:,1]

            kl_div = kl_div_bernoulli(torch.exp(log_marginals_norm[:,mdim:(mdim+bits_info[i]),1]), (torch.ones(logpm1.shape)*prior_m).to(x_flat.device))
            kl_total += kl_div
            kl_list.append(torch.sum(kl_div, dim=0)/N)

            # Re-compute the marginals of the polar-coded bits
            logqmx1 = torch.logsumexp(torch.stack((torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,0], A[i-1])+log_marginals_info_norm[:,:,1], torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,1], A[i-1])+log_marginals_info_norm[:,:,0]), dim=2), dim=2) 
            logqmx0 = torch.logsumexp(torch.stack((torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,0], A[i-1])+log_marginals_info_norm[:,:,0], torch.matmul(log_marginals_norm[:,(mdim-bits_info[i-1]):mdim,1], A[i-1])+log_marginals_info_norm[:,:,1]), dim=2), dim=2) 

            logq_marginals_mx = torch.stack((logqmx0, logqmx1), dim=2)
            logq_marginals_mx_norm = logq_marginals_mx - torch.logsumexp(logq_marginals_mx, dim=-1, keepdim=True)
            # Sanity check
            assert torch.any(torch.sum(torch.exp(logq_marginals_mx_norm), dim=-1)>(1+1e-4))==False, "Invalid probs value (sum>1)."
            assert torch.any(torch.sum(torch.exp(logq_marginals_mx_norm), dim=-1)<(1-1e-4))==False, "Invalid probs value (sum<1)."
            
            log_qc_polar[:,mdim:(mdim+bits_info[i])] = logq_marginals_mx_norm[:,:,1]
  
        else:
            
            # Compute marginals of the information bits
            logpm1 = torch.matmul(torch.log(probs[:,cdim:(cdim+H[i].shape[0])]), H[i])
            logpm0 = torch.matmul(torch.log(1-probs[:,cdim:(cdim+H[i].shape[0])]), H[i])

            log_marginals_info = torch.stack((logpm0, logpm1), dim=2)
            log_marginals_info_norm = log_marginals_info - torch.logsumexp(log_marginals_info, dim=-1, keepdim=True)

            # Sanity check
            assert torch.any(torch.sum(torch.exp(log_marginals_info_norm), dim=-1)>(1+1e-4))==False, "Invalid probs value (sum>1)."
            assert torch.any(torch.sum(torch.exp(log_marginals_info_norm), dim=-1)<(1-1e-4))==False, "Invalid probs value (sum<1)."

            log_marginals_norm[:,mdim:(mdim+bits_info[i]),0] = log_marginals_info_norm[:,:,0]
            log_marginals_norm[:,mdim:(mdim+bits_info[i]),1] = log_marginals_info_norm[:,:,1]

            kl_div = kl_div_bernoulli(torch.exp(log_marginals_norm[:,mdim:(mdim+bits_info[i]),1]), (torch.ones(logpm1.shape)*prior_m).to(x_flat.device))
            kl_total += kl_div
            kl_list.append(torch.sum(kl_div, dim=0)/N)

            # Now we do not have to recompute the marginals of the polar-coded bits!!
            log_qc_polar[:,mdim:(mdim+bits_info[i])] = log_marginals_norm[:,mdim:(mdim+bits_info[i]),1]

        qc_rep = torch.cat((qc_rep, torch.matmul(torch.exp(log_qc_polar[:,mdim:(mdim+bits_info[i])]), G[i])), dim=1)

        cdim += H[i].shape[0]
        mdim += bits_info[i]


    qz_sample = sample_from_qz_given_x(qc_rep, beta=beta, n_samples=n_samples)

    # Compute the reconstruction term E_{q(z|x)}[log p(x|z)]
    reconstruction_sum = 0

    for n in range(n_samples):

        # Forward decoder
        out_decoder = decoder.forward(qz_sample[:,:,n]).view(-1, x_flat.shape[1])

        assert torch.any(torch.isinf(out_decoder))==False, "Invalid out_decoder value (inf)."
        assert torch.any(torch.isnan(out_decoder))==False, "Invalid out_decoder value (nan)."

        # Binary observation model
        if likelihood.lower() == 'ber':
            reconstruction_sum += log_bernoulli(out_decoder, x_flat)
        # Real observation model
        elif likelihood.lower() == 'gauss':
            covar = torch.ones(out_decoder.shape[1]).to(x_flat.device) * 0.1
            reconstruction_sum += log_gaussian(x_flat, out_decoder, covar)    # Fixed variance

    reconstruction = reconstruction_sum/n_samples

    # Obtain the ELBO loss
    elbo = torch.sum((reconstruction-kl_total), dim=0)/N


    return elbo, kl_list, torch.sum(reconstruction, dim=0)/N






    

    

