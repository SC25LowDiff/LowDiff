import torch
import torch.distributed as dist
import torch.nn as nn

class TopKCompressor:
    def __init__(self, model, k=0.01):
        """
        Initialize TopKCompressor with compression ratio k.
        """
        self.k = k
        self.model = model
        self.hooked_params = []  # Marked as hooked

    def topk_compress(self, tensor):
        """
        Compress the gradient into Top-K format.
        Args:
            tensor (torch.Tensor): The gradient tensor to compress.
        Returns:
            indices (torch.Tensor): Top-K indices.
            values (torch.Tensor): Top-K values.
        """
        num_elements = tensor.numel()
        k_elements = max(1, int(num_elements * self.k))

        # Get Top-K indices and values
        values, indices = torch.topk(tensor.view(-1).abs(), k_elements, sorted=False)
        values = tensor.view(-1).gather(0, indices)

        return indices, values

    def _hook_fn(self, grad):             
        # Perform Top-K compression
        indices, values = self.topk_compress(grad)

        # Perform all_gather communication
        world_size = dist.get_world_size()

        # Allocate receive buffers
        gathered_indices = [torch.zeros_like(indices) for _ in range(world_size)]
        gathered_values = [torch.zeros_like(values) for _ in range(world_size)]

        # Perform communication
        dist.all_gather(gathered_indices, indices, async_op=True)
        dist.all_gather(gathered_values, values, async_op=True)

        # Restore the gradient
        restored_grad = torch.zeros_like(grad).view(-1)

        for idx, val in zip(gathered_indices, gathered_values):
            restored_grad.scatter_add_(0, idx, val)
        
        # Return the restored gradient
        return restored_grad.view_as(grad)
    
    def register_hooks(self):
        """
        Register Top-K compression hooks for parameters in the model.
        Args:
            model (nn.Module): The model to register hooks.
        """
        self.hooked_params = []  # Clear the list to avoid duplicate entries

        for layer in self.model.modules():
            # Register a gradient hook for each parameter
            for param in layer.parameters():
                if param.requires_grad:
                    # Register a gradient hook
                    param.register_hook(lambda grad, param=param: self._hook_fn(grad))

                    # Store parameters of the hooked layer
                    self.hooked_params.append(param)

    def freeze_hooked_params(self):
        """Freeze parameters that have been hooked to prevent gradient updates."""
        for param in self.hooked_params:
            param.requires_grad = False

    def unfreeze_hooked_params(self):
        """Unfreeze parameters to allow gradient updates."""
        for param in self.hooked_params:
            param.requires_grad = True