# config.py

from torchvision import transforms

# Common normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
}

# Dataset directories
dataset_dirs = {
    'scisic': {
        'train': 'scisic/Train',
        'test': 'scisic/Test'
    },
    'ccts': {
        'train': 'ccts/train',
        'val': 'ccts/valid',
        'test': 'ccts/test'
    },
    'rotc': {
        'train': 'rotc/train',
        'val': 'rotc/val',
        'test': 'rotc/test'
    }
}


