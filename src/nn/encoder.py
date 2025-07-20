import torch
from torch import nn
from src.nn.modules import LambdaLayer, ScaledTanh
from src.nn.modules import dclamp

class Encoder(nn.Module):
    """
    General class implementing the decoder of the model.
    """

    def __init__(self, enc, inference_type='uncoded'):
        """
        Initialize an instance of the class.

        Parameters
        ----------
        enc: torch.nn.Module
            Module with the architecture of the encoder neural network without the output activation.
        """

        super(Encoder, self).__init__()

        self.enc = enc
        self.inference_type = inference_type
        
        # The encoder outputs bit probabilities q(bit=1|x)
        self.enc.append(nn.Sigmoid())


    def forward(self, x):

        """
        Forward pass.

        Parameters
        ----------
        x: torch.tensor
            Batch of data.
        """

        # Forward the encoder
        out = self.enc(x)
        # Clamp the output to avoid numerical instabilities
        out = dclamp(out, 0.001, 0.999) 

        return out


