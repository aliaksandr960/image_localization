import torch
import torch.nn as nn
import torch.optim as optim
import torchgeo

from triplet_datasets.augmentation import make_train_aug
from triplet_datasets.torchgeo_adapters import filter_tg_indexes_by_max_change
from triplet_datasets.datasets import LocalTripletRandomDataset, LocalTripletStaticDataset

from siam_baseline.models import TimmMobilenet
from siam_baseline.training_pipeline import run_trining

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup model
net = TimmMobilenet('mobilenetv3_small_100').to(device)

# Setup training params
batch_size = 16
epoch_count = 2
epoch_max_batch = 64
num_workers = 2
patch_size = 512, 512

criterion = nn.TripletMarginLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# Setup trainig dataset
levircdplus_tg_train = torchgeo.datasets.LEVIRCDPlus(
    root='data/levircdplus_train',
    split='train',
    transforms=None,
    download=True,
    checksum=False
)
change_threshold = 0.025
train_index_list = filter_tg_indexes_by_max_change(levircdplus_tg_train, change_threshold)


train_dataset = LocalTripletRandomDataset(levircdplus_tg_train,
                                        train_index_list,
                                        augmentations=make_train_aug(patch_size))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers)

# Setup valid dataset
levircdplus_tg_valid = torchgeo.datasets.LEVIRCDPlus(
    root='data/levircdplus_train',
    split='test',
    transforms=None,
    download=True,
    checksum=False
)
valid_index_list = filter_tg_indexes_by_max_change(levircdplus_tg_valid, 0.025)
valid_dataset = LocalTripletStaticDataset(levircdplus_tg_valid,
                                        valid_index_list)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1,
                                        shuffle=False, num_workers=num_workers)

# Setup logging
train_history_path = './working/t_history.csv'
valid_history_path = './working/v_history.csv'
log_path = './working/'

# Run training
run_trining(net=net,
            criterion=criterion,
            optimizer=optimizer,
            epoch_count=epoch_count,
            epoch_max_batch=epoch_max_batch,
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_history_path=train_history_path,
            valid_history_path=valid_history_path,
            log_path=log_path,
            device=device
            )