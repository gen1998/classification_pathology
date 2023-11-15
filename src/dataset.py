import pandas as pd

import torch
from torch.utils.data import Dataset
from typing import Tuple, List

from src.augmentation import get_valid_transforms, get_train_transforms
from src.utils import get_img

class HepaDataset(Dataset):
    """Pytorch HepaDataset
    """
    def __init__(self, df: pd.DataFrame, transforms=None):
        """
        Args:
            df (pd.DataFrame): columns are label and img_path
            transforms (optional): Augmentation Function. Defaults to None.
        """
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        target = self.df.loc[index]["label"]

        img  = get_img(self.df.loc[index]["img_path"])

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, torch.tensor(target)

def set_train_dataloader(df: pd.DataFrame, input_shape: int, train_bs: int, valid_bs: int, num_workers: int,
                         trn_idx: List[int]=None, val_idx: int=None):
    """Prepare train and valid dataloaders with the use of fold

    Args:
        df (pd.DataFrame): include only img_path and label
        input_shape (int): input image shape
        train_bs (int): train batch size
        valid_bs (int): validation batch size
        num_workers (int): num_workers
        
        trn_idx(list): if you use fold, it is train fold list
        val_idx(int): if you use fold, it is validate fold

    Returns:
        train_loader: train loader
        val_loader: validate loader
    """

    train_ = df[df.fold.isin(trn_idx)].reset_index(drop=True)
    valid_ = df[df.fold==val_idx].reset_index(drop=True)

    train_ds = HepaDataset(train_, transforms=get_train_transforms(input_shape))
    valid_ds = HepaDataset(valid_, transforms=get_valid_transforms(input_shape))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_bs,
        pin_memory=True, # faster and use memory
        drop_last=False,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=valid_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader

def set_infer_dataloader(df: pd.DataFrame, input_shape: int, valid_bs: int, num_workers: int, val_idx: int=None, tst_idx: int=None):
    """Prepare infer dataloder as well as set_train_dataloder
    """
    valid_ = df[df["fold"]==val_idx].reset_index(drop=True)
    test_ = df[df["fold"]==tst_idx].reset_index(drop=True)

    valid_ds = HepaDataset(valid_, transforms=get_valid_transforms(input_shape))
    test_ds = HepaDataset(test_, transforms=get_valid_transforms(input_shape))

    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=valid_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    tst_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=valid_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )

    return tst_loader, test_, val_loader, valid_

def set_kmeans_dataloader(df, input_shape, tst_idx, val_idx, valid_bs, num_workers, split: bool=True):
    """Prepare kmeans dataloader as well as set_train_dataloader
    """
    if split:
        test_ = df[df.split==2].reset_index(drop=True)
        val_ = df[df.split==1].reset_index(drop=True)
    else:
        test_ = df[df["fold"]==tst_idx].reset_index(drop=True)
        val_ = df[df["fold"]==val_idx].reset_index(drop=True)

    test_ds = HepaDataset(test_, transforms=get_valid_transforms(input_shape))
    val_ds = HepaDataset(val_, transforms=get_valid_transforms(input_shape))

    tst_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=valid_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=valid_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )

    return tst_loader, test_, val_loader, val_

def sur_6fold_dataloader(df, input_shape, valid_bs, num_workers):
    test_ = df[df["fold"]==6].reset_index(drop=True)
    test_ds = HepaDataset(test_, transforms=get_valid_transforms(input_shape))

    tst_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=valid_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )

    return tst_loader, test_