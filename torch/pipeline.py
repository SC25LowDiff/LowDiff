import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import torch
import torch.distributed as dist
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import deepspeed
from deepspeed.pipe import PipelineModule
import pipelineengine

parser = argparse.ArgumentParser(description='CIFAR')
parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
parser.add_argument('-s', '--steps', type=int, default=1000, help='quit after this many steps')
parser.add_argument('-p', '--pipeline-parallel-size', type=int, default=2, help='pipeline parallelism')
parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
parser.add_argument('--seed', type=int, default=42, help='PRNG seed')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers

def main():
    deepspeed.init_distributed()
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(args.seed)
    print(f"[Rank {rank}/{world_size}] Initialized DeepSpeed.")

    dataset = datasets.ImageFolder(
            '/data/dataset/cv/imagenet_0908/train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        )
    
    model = models.vgg19_bn()
    
    model = PipelineModule(layers=join_layers(model),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         activation_checkpoint_interval=0)

    ds_config = {
        "train_batch_size" : 256,
        "train_micro_batch_size_per_gpu" : 8,
        "optimizer": {
            "type": "SGD",
            "params": {
            "lr": 0.001
            }
        },
        "steps_per_print" : 1
    }
    model, _, _, _ = pipelineengine.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config)

    model.pipeline_enable_backward_allreduce = False
    
    for step in range(args.steps):
        loss = model.train_batch()
    
if __name__ == '__main__':
    main()
    
