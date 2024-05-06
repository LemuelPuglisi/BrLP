import torch
from torch import Tensor


class KLDivergenceLoss:
    """
    A class for computing the Kullback-Leibler divergence loss.
    """
    
    def __call__(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        """
        Computes the KL divergence loss for the given parameters.

        Args:
            z_mu (Tensor):  The mean of the distribution.
            z_sigma (Tensor): The standard deviation of the distribution.

        Returns:
            Tensor: The computed KL divergence loss, averaged over the batch size.
        """

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]