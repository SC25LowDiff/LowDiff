import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import deepspeed
from deepspeed import comm as dist
from communicator.checkfreq import Communicator
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)
import copy

# Argument parsing
parser = argparse.ArgumentParser(description='DeepSpeed ImageNet Training with TopK Compression')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
parser.add_argument('--model', default='resnet101', type=str, help='model architecture')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='batch size per GPU')
parser.add_argument('--lr', '--learning-rate', default=0.0125, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--workers', default=1, type=int, help='data loading workers')
parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
parser.add_argument("--save-dir", default='save_dir', type=str, help='directory to save checkpoints')
parser.add_argument("--resume", type=int, default=0, help='resume from checkpoint')
parser.add_argument("--freq", default=0, type=int, help='how many iteration to save a full checkpoint')

args = parser.parse_args()

def main():
    # Initialize DeepSpeed
    deepspeed.init_distributed()
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
                transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.267, 0.256, 0.276])
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
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # optionally resume from a checkpoint at rank 0, then broadcast weights to other workers
    if args.resume and dist.get_rank() == 0:
        model, optimizer = load_base_checkpoint(model,optimizer)
    
    # Define the configuration dictionary directly in code
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1
    }
    
    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, model_parameters=model.parameters(), config=ds_config)
    model.enable_backward_allreduce = False
    
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

            if dist.get_rank() == 0:
                print("[Epoch {}/{}] Batch {}, Loss: {:.3f}, Time: {:.3f}"
                    .format(epoch, args.epochs, batch_idx, loss.item(), time.time() - end))
            
            if dist.get_rank() == 0 and args.freq > 0 and batch_idx % args.freq == 0:
                # snapshot
                begin = time.time()
                model_snapshot = snapshot(model)
                end = time.time()
                print("snapshot takes {:.3f}s".format(end - begin))
                # persist
                communicator.queue.put((model_snapshot,'{}/{}_{}_{}_full.pth.tar'.format(args.save_dir,args.model,epoch,batch_idx)))

            end = time.time()

        print(f"Epoch {epoch} completed.")

def load_base_checkpoint(model, optimizer):
    start = time.time()
    filedir = args.save_dir
    filepath = filedir + '/' + args.model + '_' + args.dataset + '_' + args.compressor + '_' + str(args.compressor_ratio) + '_' + str(args.resume-1) + '_0_full' + '.pth.tar'
    if os.path.isfile(filepath):
        print("loading {}".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        end = time.time()
        print("load base checkpoint takes {:.3f}s".format(end - start))
        return model, optimizer
    else:
        raise ValueError("No checkpoint found")

def topk_decompress(values, indices, shape):
    tensor_decompressed = torch.zeros(shape).cuda().view(-1)
    for idx, val in zip(indices, values):
        tensor_decompressed = tensor_decompressed.scatter_add_(0, idx, val)
    return tensor_decompressed.view(shape)

def _to_cpu(data):
    """
    Move tensor to CPU and return
    """
    if hasattr(data, 'cpu'):
        return data.detach().cpu().clone()
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(v) for v in data)
    else:
        return data

def snapshot(model):
    model_snapshot = copy.deepcopy(model.module).cpu()
    optimizer_snapshot = _to_cpu(copy.deepcopy(model.optimizer.state_dict()))
    return [model_snapshot, optimizer_snapshot]
    
if __name__ == '__main__':
    main()
