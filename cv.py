import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import deepspeed
from compression.comm import Communicator

# Argument parsing
parser = argparse.ArgumentParser(description='DeepSpeed ImageNet Training with TopK Compression')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
parser.add_argument('--model', default='resnet101', type=str, help='model architecture')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='batch size per GPU')
parser.add_argument('--lr', '--learning-rate', default=0.0125, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--workers', default=8, type=int, help='data loading workers')
parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
parser.add_argument('--compress_ratio', default=0.01, type=float, help='TopK compression ratio')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
parser.add_argument("--compressor", default="topk", type=str, help='which compressor to use')
parser.add_argument("--compressor_ratio", default=0.01, type=float, help='choose compress ratio for compressor')
parser.add_argument("--diff", default=False, type=bool, help='whether to use differentail ckpt')
parser.add_argument("--freq", default=0, type=int, help='how many iteration to save a whole checkpoint')
parser.add_argument("--pack", default='1', type=int, help='in-memory packing frequency')
args = parser.parse_args()

def main():
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(args.seed)
    print(f"[Rank {rank}/{world_size}] Initialized DeepSpeed.")

    # Load dataset
    if args.dataset == 'imagenet':
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

    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(
            '/data/dataset/cv/cifar100/train',
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            ])
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers)

    # Model selection
    if args.model == 'resnet50':
        model = models.resnet50()
    elif args.model == 'resnet101':
        model = models.resnet101()
    elif args.model == 'vgg16':
        model = models.vgg16_bn()
    elif args.model == 'vgg19':
        model = models.vgg19_bn()
    else:
        print("Model ERROR!")
        return
    
    model.cuda()
    
    # Initialize DeepSpeed
    model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config='ds_config.json')
    
     # Use the Communicator class
    communicator = Communicator(model)
    communicator.register_hooks()
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, (images, targets) in enumerate(train_loader):
            end = time.time()
            images, targets = images.cuda(), targets.cuda()
            
            output = model(images)
            loss = criterion(output, targets)
            model.backward(loss)
            communicator.decompress()
            model.step()

            if deepspeed.comm.get_rank() == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Batch {batch_idx}, Loss: {loss.item()}, Time: {time.time() - end}")

            end = time.time()

        print(f"Epoch {epoch} completed.")


if __name__ == '__main__':
    main()
