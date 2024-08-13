from typing import Optional

from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler


class GradientAccumulation:
    """
    Implements gradient accumulation to facilitate training with larger 
    effective batch sizes than what can be physically accommodated in memory.
    """

    def __init__(self,
                 actual_batch_size: int, 
                 expect_batch_size: int,
                 loader_len: int,
                 optimizer: Optimizer, 
                 grad_scaler: Optional[GradScaler] = None) -> None:
        """
        Initializes the GradientAccumulation instance with the necessary parameters for 
        managing gradient accumulation.

        Args:
            actual_batch_size (int): The size of the mini-batches actually used in training.
            expect_batch_size (int): The desired (effective) batch size to simulate through gradient accumulation.
            loader_len (int): The length of the data loader, representing the total number of mini-batches.
            optimizer (Optimizer): The optimizer used for performing optimization steps.
            grad_scaler (Optional[GradScaler], optional): A GradScaler for mixed precision training. Defaults to None.
        
        Raises:
            AssertionError: If `expect_batch_size` is not divisible by `actual_batch_size`.
        """

        assert expect_batch_size % actual_batch_size == 0, \
            'expect_batch_size must be divisible by actual_batch_size'
        self.actual_batch_size = actual_batch_size
        self.expect_batch_size = expect_batch_size
        self.loader_len = loader_len
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler

        # if the expected batch size is N=KM, and the actual batch size
        # is M, then we need to accumulate gradient from N / M = K optimization steps. 
        self.steps_until_update = expect_batch_size / actual_batch_size

    def step(self, loss: Tensor, step: int) -> None:
        """
        Performs a backward pass for the given loss and potentially executes an optimization 
        step if the conditions for gradient accumulation are met. The optimization step is taken 
        only after a specified number of steps (defined by the expected batch size) or at the end 
        of the dataset.

        Args:
            loss (Tensor): The loss value for the current forward pass.
            step (int): The current step (mini-batch index) within the epoch.
        """
        loss = loss / self.expect_batch_size
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()        
        if (step + 1) % self.steps_until_update == 0 or (step + 1) == self.loader_len:
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)