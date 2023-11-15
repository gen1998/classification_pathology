import sys
sys.path.append("pytorch-image-models/")

import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

try:
    from cuml.feature_extraction.text import TfidfVectorizer
    from cuml.neighbors import NearestNeighbors
    from cuml.cluster import KMeans as KMeans
except:
    pass

import timm
import math
import torch.nn.functional as F

import lightgbm as lgb
from abc import abstractmethod


class HepaClassifier(nn.Module):
    """Classification Model
    """
    def __init__(self, model_arch: str, n_class: int, model_shape: str, pretrained: bool=True):
        """
        Args:
            model_arch (str): model architecture name
            n_class (int): number of classification
            model_shape (str): Due to the model's structure, the connection layers change.
            pretrained (bool, optional): pretrained model or not. Defaults to True.
        """
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)

        if model_shape == "eff":
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
        elif model_shape == "vit":
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, n_class)
        elif model_shape == "res":
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x

class EarlyStopping:
    """Early Stopping Class
    """
    def __init__(self, patience: int):
        """
        Args:
            patience (int): How many epochs should be tolerated without seeing a value update
        """
        self.max_val_monitor = 1000
        self.min_val_monitor = -1000
        self.val_epoch = -1
        self.stop_count = 0
        self.patience = patience
        self.min_delta = 0

    # mode = "min" or "max"(val_loss, val_accuracy)
    def update(self, monitor, epoch, mode):
        if mode == "max":
            if monitor > self.min_val_monitor:
                self.min_val_monitor = monitor
                self.val_epoch = epoch
                self.stop_count = 0
            else:
                self.stop_count+=1
        else:
            if monitor < self.max_val_monitor:
                self.max_val_monitor = monitor
                self.val_epoch = epoch
                self.stop_count = 0
            else:
                self.stop_count+=1

        if self.stop_count >= self.patience:
            return -1
        else:
            return 0

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, verbose_step, scheduler=None, schd_batch_update=False):
    """ train one epoch
    """
    model.train()
    scaler = GradScaler()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        with autocast():
            image_preds = model(imgs)
            loss = loss_fn(image_preds, image_labels)

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None and schd_batch_update:
                scheduler.step()

            if ((step + 1) % verbose_step == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)

    print("train: "+ description)
    if scheduler is not None and not schd_batch_update:
        scheduler.step()

def valid_one_epoch(epoch, model, loss_fn, val_loader, device, verbose_step, scheduler=None, schd_loss_update=False):
    """valid one epoch
    """
    model.eval()
    
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        image_preds = model(imgs)

        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % verbose_step== 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)

    print("valid "+ description)
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()

    monitor = {}
    monitor["val_loss"] = loss_sum/sample_num
    monitor["val_accuracy"] = (image_preds_all==image_targets_all).mean()
    return monitor

def inference_one_epoch(model, data_loader, device):
    """inference one epochs
    """
    model.eval()

    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs, _) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

def inference_embeddings(model, data_loader, device):
    """extract embeddings
    """
    model.eval()
    image_preds_all = []
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook


    for step, (imgs, _) in pbar:
        imgs = imgs.to(device).float()

        model.model.global_pool.register_forward_hook(get_activation('act2'))
        image_preds_all += [torch.softmax(activation["act2"], 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    print(image_preds_all.shape)
    return image_preds_all

def get_image_predictions_kmeans(df_tst, df_val, test_embeddings, val_embeddings, n_clusters, index):
    """train and predict cluster
    val : train
    test : predict
    """
    if index == 0:
        model = KMeans(n_clusters=n_clusters, max_iter=300, init='scalable-k-means++')
        model.fit(val_embeddings)
        df_val[f"labels_{index}"] = model.labels_
        df_tst[f"labels_{index}"] = model.predict(test_embeddings)
    else:
        df_val[f"labels_{index}"] = -100
        df_tst[f"labels_{index}"] = -100
        indexes = df_val[f"labels_{index-1}"].unique()
        for i in indexes:
            buff = df_val[df_val[f"labels_{index-1}"]==i]
            if len(buff) < 10 or i==-100:
                continue
            p = len(buff[buff["label"]==0])/len(buff)
            if p>0.95 or p<0.05:
                continue
            attention_index_val = np.where(df_val[f"labels_{index-1}"]==i)
            attention_index_tst = np.where(df_tst[f"labels_{index-1}"]==i)
            embeddings_val = val_embeddings[attention_index_val]
            embeddings_tst = test_embeddings[attention_index_tst]
            if len(embeddings_val) < 1:
                continue
            model = KMeans(n_clusters=n_clusters, max_iter=300, init='scalable-k-means++')
            model.fit(embeddings_val)
            df_val.loc[attention_index_val[0], f"labels_{index}"] = model.labels_ + 2*i
            if len(embeddings_tst) < 1:
                continue
            df_tst.loc[attention_index_tst[0], f"labels_{index}"] = model.predict(embeddings_tst) + 2*i

    return df_val, df_tst
