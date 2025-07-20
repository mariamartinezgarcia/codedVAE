import torch
import wandb
from src.train.loss import get_elbo_uncoded, get_elbo_rep, get_elbo_hier

def has_nan_or_inf(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def train_step(model, x, inference=None, n_samples=1, train_enc=True, train_dec=True):

    """
    Train step.
        
        Parameters
        ----------
        model : CodedDVAE instance
            Model to be trained.
        x : torch.tensor
            Batch of data.
        inference: string
            Inference type.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
            - 'hier' for the coded hierarchical case.
        n_sampes : int, optional
            Number of samples used for computing the ELBO. The number of samples is 1 by default.
        train_enc : boolean, optional
            Flag to indicate if the parameters of the encoder need to be updated. True by default.
        train_enc : boolean, optional
            Flag to indicate if the parameters of the decoder need to be updated. True by default.
        
        Returns
        -------
        elbo : torch.tensor
            Value of the ELBO.
        kl_div : torch.tensor
            Value of the Kullback-Leibler divergence term in the ELBO. In the hierarchical case, the method returns a list with two torch.tensor elements containing the two KL Divergence terms.
        reconstruction: torch.tensor
            Value of the reconstruction term in the ELBO.
    """

    x = x.to(model.device)

    # Neural Networks in training mode
    model.encoder.train()
    model.decoder.train()

    model.optimizer_encoder.zero_grad()
    model.optimizer_decoder.zero_grad()

    # Uncoded case
    if inference == 'uncoded':

        # Compute loss
        elbo, kl_div, reconstruction = get_elbo_uncoded(x, model.encoder, model.decoder, beta=model.beta, likelihood=model.likelihood, n_samples=n_samples)

        # Sanity check
        assert torch.any(torch.isinf(elbo))==False, "Invalid ELBO value (inf)."
        assert torch.any(torch.isnan(elbo))==False, "Invalid ELBO value (nan)."

        # Gradients
        loss = -elbo
        loss.backward()

        # Optimizer step
        if train_dec:
            model.optimizer_decoder.step()
        if train_enc:
            model.optimizer_encoder.step()

    # Coded case (repetition code)
    if inference == 'rep':

        # Compute loss
        elbo, kl_div, reconstruction = get_elbo_rep(x, model.encoder, model.decoder, model.G, model.H, beta=model.beta, likelihood=model.likelihood, n_samples=n_samples)

        # Sanity check
        assert torch.any(torch.isinf(elbo))==False, "Invalid ELBO value (inf)."
        assert torch.any(torch.isnan(elbo))==False, "Invalid ELBO value (nan)."

        # Gradients
        loss = -elbo
        loss.backward()

        # Optimizer step
        if train_dec:
            model.optimizer_decoder.step()
        if train_enc:
            model.optimizer_encoder.step()

    if inference == 'hier':

        # Compute loss
        elbo, kl_div, reconstruction = get_elbo_hier(x, model.encoder, model.decoder, model.G, model.H, beta=model.beta, A=model.A, bits_info=model.bits_info, bits_info_total=model.bits_info_total, likelihood=model.likelihood, n_samples=n_samples, polar=model.polar)

        # Sanity check
        assert torch.any(torch.isinf(elbo))==False, "Invalid ELBO value (inf)."
        assert torch.any(torch.isnan(elbo))==False, "Invalid ELBO value (nan)."

        # Gradients
        loss = -elbo
        loss.backward()

        # Optimizer step
        if train_dec:
            model.optimizer_decoder.step()
        if train_enc:
            model.optimizer_encoder.step()
    
    return elbo, kl_div, reconstruction


def val_step(model, x, inference=None, n_samples=1):

    """
    Validation step.
        
        Parameters
        ----------
        model : CodedDVAE instance
            Model to be trained.
        x : torch.tensor
            Batch of data.
        inference: string
            Inference type.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
            - 'hier' for the coded hierarchical case.
        n_sampes : int, optional
            Number of samples used for computing the ELBO. The number of samples is 1 by default.
        
        Returns
        -------
        elbo : torch.tensor
            Value of the ELBO.
        kl_div : torch.tensor
            Value of the Kullback-Leibler divergence term in the ELBO. In the hierarchical case, the method returns a list with two torch.tensor elements containing the two KL Divergence terms.
        reconstruction: torch.tensor
            Value of the reconstruction term in the ELBO.
    """

    x = x.to(model.device)

    # Neural Networks in eval mode
    model.encoder.eval()
    model.decoder.eval()

    with torch.no_grad():
        # Uncoded case
        if inference == 'uncoded':

            # Compute loss
            elbo, kl_div, reconstruction = get_elbo_uncoded(x, model.encoder, model.decoder, beta=model.beta, likelihood=model.likelihood, n_samples=n_samples)

            # Sanity check
            assert torch.any(torch.isinf(elbo))==False, "Invalid ELBO value (inf)."
            assert torch.any(torch.isnan(elbo))==False, "Invalid ELBO value (nan)."

        # Coded case (repetition code)
        if inference == 'rep':

            # Compute loss
            elbo, kl_div, reconstruction = get_elbo_rep(x, model.encoder, model.decoder, model.G, model.H, beta=model.beta, likelihood=model.likelihood, n_samples=n_samples)

            # Sanity check
            assert torch.any(torch.isinf(elbo))==False, "Invalid ELBO value (inf)."
            assert torch.any(torch.isnan(elbo))==False, "Invalid ELBO value (nan)."

        if inference == 'hier':

            # Compute loss
            elbo, kl_div, reconstruction = get_elbo_hier(x, model.encoder, model.decoder, model.G, model.H, beta=model.beta, A=model.A, bits_info=model.bits_info, bits_info_total=model.bits_info_total, likelihood=model.likelihood, n_samples=n_samples, polar=model.polar)

            # Sanity check
            assert torch.any(torch.isinf(elbo))==False, "Invalid ELBO value (inf)."
            assert torch.any(torch.isnan(elbo))==False, "Invalid ELBO value (nan)."

    
    return elbo, kl_div, reconstruction

    
    
def trainloop(model, train_dataloader, n_epochs, inference=None, n_samples=1, train_enc=True, train_dec=True, verbose=True, wb=False, start_epoch=0, val_dataloader=None, freq_valid=5):

    """
    Trainloop to train the model for a given number of epochs.
        
        Parameters
        ----------
        model : CodedDVAE instance
            Model to be trained.
        train_dataloader : torch Dataloader
            Dataloader with the training set.
        n_epochs: int
            Number of epochs.
        inference: string
            Inference type.
            - 'uncoded' for the uncoded case.
            - 'rep' for the coded case with inference at bit level using repetition codes.
            - 'hier' for the coded hierarchical case.
        n_sampes : int, optional
            Number of samples used for computing the ELBO. The number of samples is 1 by default.
        train_enc : boolean, optional
            Flag to indicate if the parameters of the encoder need to be updated. True by default.
        train_enc : boolean, optional
            Flag to indicate if the parameters of the decoder need to be updated. True by default.
        verbose: boolean, optional
            Flag to print the ELBO during training. True by default.
        wb: boolean, optional
            Flag to log the ELBO, KL term and reconstruction term to Weights&Biases.
        start_epoch: int, optional
            Epoch where the trainloop starts. This is useful to obtain coherent logs in weights and biases when we finetune a model.
        val_dataloader : torch Dataloader, optional
            Dataloader with the validation set.
        Returns
        -------
        elbo_evolution : list
            List containing the ELBO values obtained during training (1 value per epoch).
        kl_div_evolution : list
            List containing the Kullback-Leibler divergence values obtained during training (1 value per epoch).
        reconstruction_evolution : list
            List containing reconstruction term values obtained during training (1 value per epoch).
    """

    elbo_evolution = []
    kl_evolution = []
    rec_evolution = []

    elbo_evolution_val = []
    kl_evolution_val = []
    rec_evolution_val = []

    for e in range(start_epoch, n_epochs):

        elbo_epoch = 0
        kl_epoch = 0
        if inference == 'hier':
            kl_list = [0]*len(model.bits_info)
        reconstruction_epoch = 0

        for x, _ in train_dataloader:    # Batches
            
            elbo, kl, reconstruction = train_step(model, x, inference=inference, n_samples=n_samples, train_enc=train_enc, train_dec=train_dec)

            elbo_epoch += elbo.item()
            reconstruction_epoch += reconstruction.item()
            
            if inference=='hier':

                kl_list = [kl_list[i]+kl[i].item() for i in range(len(kl))]


            else:
                kl_epoch += kl.item()
            

        elbo_evolution.append(elbo_epoch/len(train_dataloader))     
        rec_evolution.append(reconstruction_epoch/len(train_dataloader)) 

        if inference=='hier':
            kl_epoch = [x/len(train_dataloader) for x in kl_list]
            kl_evolution.append(kl_epoch)

        else: 
            kl_epoch = kl_epoch/len(train_dataloader)
            kl_evolution.append(kl_epoch)


        if not(val_dataloader is None) and (e%freq_valid == 0):
            
            elbo_val_epoch = 0
            kl_val_epoch = 0
            if inference == 'hier':
                kl_val_list = [0]*len(model.bits_info)
            reconstruction_epoch = 0
            for x, _ in val_dataloader: 
                elbo_val, kl_val, reconstruction_val = val_step(model, x, inference=inference, n_samples=n_samples)
                elbo_val_epoch += elbo_val.item()
                reconstruction_epoch_val += reconstruction_val.item()
                
                if inference=='hier':
                    kl_val_list = [kl_val_list[i]+kl_val[i].item() for i in range(len(kl_val))]
                else:
                    kl_val_epoch += kl_val.item()

            elbo_evolution_val.append(elbo_val_epoch/len(val_dataloader))     
            rec_evolution_val.append(reconstruction_epoch/len(val_dataloader)) 

            if inference=='hier':
                kl_val_epoch = [x/len(val_dataloader) for x in kl_val_list]
                kl_evolution_val.append(kl_val_epoch)

            else: 
                kl_val_epoch = kl_val_epoch/len(val_dataloader)
                kl_evolution_val.append(kl_val_epoch)

            if wb:  
                if inference == 'hier':
                    wandb.log({"elbo/epoch": elbo_epoch/len(train_dataloader),
                                "reconstruction/epoch": reconstruction_epoch/len(train_dataloader),
                                "elbo_val/epoch": elbo_val/len(val_dataloader),
                                "reconstruction_val/epoch": reconstruction_epoch_val/len(val_dataloader),
                                "epoch:": e })
                    for i in range(len(kl_epoch)):
                        wandb.log({"kl"+str(i+1): kl_epoch[i],
                                   "kl"+str(i+1)+"_val": kl_val_epoch[i],
                                    "epoch:": e })
                    
                else:
                    wandb.log({"elbo/epoch": elbo_epoch/len(train_dataloader),
                                "kl/epoch":kl_epoch,
                                "reconstruction/epoch": reconstruction_epoch/len(train_dataloader),
                                "elbo_val/epoch": elbo_val_epoch/len(val_dataloader),
                                "kl_val/epoch":kl_val_epoch,
                                "reconstruction_val/epoch": reconstruction_epoch/len(val_dataloader),
                                "epoch:": e })
        else: 

            if wb:  
                if inference == 'hier':
                    wandb.log({"elbo/epoch": elbo_epoch/len(train_dataloader),
                                "reconstruction/epoch": reconstruction_epoch/len(train_dataloader),
                                "epoch:": e })
                    for i in range(len(kl_epoch)):
                        wandb.log({"kl"+str(i+1): kl_epoch[i],
                                    "epoch:": e })
                else:
                    wandb.log({"elbo/epoch": elbo_epoch/len(train_dataloader),
                                "kl/epoch":kl_epoch,
                                "reconstruction/epoch": reconstruction_epoch/len(train_dataloader),
                                "epoch:": e })

        
        if verbose:
            print("ELBO after %d epochs: %f" %(e+1, elbo_evolution[-1]))

            if not(val_dataloader is None) and (e%freq_valid == 0):
                print("ELBO [validation] after %d epochs: %f" %(e+1, elbo_evolution_val[-1]))

        

    return elbo_evolution, kl_evolution, rec_evolution 