import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.random_erasing import RandomErasing
from dataset.retrieval_dataset import ft_train_dataset, ft_test_dataset
from dataset.pretrain_dataset import pre_dataset



def create_dataset(dataset, config, evaluate=False):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    box_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    train_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = re_test_dataset(config['test_file'], config, test_transform)
    val_dataset = re_test_dataset(config['val_file'], config, test_transform) if dataset != 'icfg' else None
    if evaluate:
        return None, val_dataset, test_dataset

    if dataset == 'sda':
        train_dataset = RegionTextJsonDataset(config, box_transform)
    else:
        train_dataset = re_train_dataset(config, train_transform)
    return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
