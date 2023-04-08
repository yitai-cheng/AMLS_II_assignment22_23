"""Constants for the project."""
from torchvision import transforms

TRAIN_BATCH_SIZE = 64
TRAIN_EPOCH_NUM = 20
MODALITIES = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, ratio=(0.8, 1.2)),
        transforms.RandomRotation(degrees=(0, 360)),
        transforms.RandomHorizontalFlip()
        # transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean, std)
    ]),

    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean, std)
    ])
}
