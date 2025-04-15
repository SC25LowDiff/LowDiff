import torch
import torch.distributed as dist

class Allgather:
    def __init__(self, compressor, world_size):
        """
        Initialize the AllgatherDS communicator.
        Args:
            compressor: The compression module.
            world_size: The number of processes in the distributed group.
        """
        self.compressor = compressor
        self.world_size = world_size

    def async_send(self, tensors_compressed):
        """
        Asynchronously send compressed tensors using all_gather.
        Args:
            tensors_compressed (list[torch.Tensor]): List of compressed tensors.
        Returns:
            Future handles and tensor sizes per rank.
        """
        # Get the size of the first dimension of each tensor
        tensors_size = [t.numel() if t.ndimension() == 0 else t.size(0) for t in tensors_compressed]

        # Exchange tensor size information
        tensors_size_tensor = torch.tensor(tensors_size, device=tensors_compressed[0].device)
        gathered_sizes = [torch.zeros_like(tensors_size_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_sizes, tensors_size_tensor)
        tensor_sizes = torch.stack(gathered_sizes).t().tolist()  # Transpose to get the rank-wise size list

        # Asynchronous all_gather operation
        futures = []
        for tensor_compressed in tensors_compressed:
            gathered_tensors = [torch.zeros_like(tensor_compressed) for _ in range(self.world_size)]
            future = dist.all_gather(gathered_tensors, tensor_compressed, async_op=True).get_future()
            futures.append((future, gathered_tensors))

        return futures, tensor_sizes

    def wait_receive(self, result, ctx):
        """
        Wait for the asynchronous communication to complete and decompress the received tensors.
        Args:
            result: The tuple returned by async_send.
            ctx: Context for decompression.
        Returns:
            Aggregated tensors.
        """
        futures, tensor_sizes = result
        tensors_ag = []

        # Wait for all futures to complete
        for future, gathered_tensors in futures:
            future.wait()
            tensors_ag.append(gathered_tensors)

        # Split the data according to the size of each rank
        tensors_ag_split = [list(torch.split(t, sizes)) for t, sizes in zip(tensors_ag, tensor_sizes)]

        # Decompress
        list_tensor_decompressed = []
        for tensor_compressed in zip(*tensors_ag_split):
            tensor_decompressed = self.compressor.decompress(tensor_compressed, ctx)
            list_tensor_decompressed.append(tensor_decompressed)

        # Normalize and return
        tensors_aggregated = self.compressor.aggregate(list_tensor_decompressed)
        return (tensors_aggregated / self.world_size) if self.compressor.average else tensors_aggregated