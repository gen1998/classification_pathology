import json
import os
import pandas as pd
import numpy as np
import torch

from src.dataset import set_train_dataloader, set_infer_dataloader
from src.utils import seed_everything
from src.model import HepaClassifier, EarlyStopping, train_one_epoch, valid_one_epoch, inference_one_epoch
from src.read_data import data_load

def main():
    # config, global value setting
    config = json.load(open('./config/default.json'))
    seed_everything(config['seed'])
    device = torch.device(config['device'])

    image_df = data_load()

    # 予測結果を保存
    folder_name = config["folder_name"]
    folder_path = f"data/output/result/{folder_name}"
    test_df = pd.DataFrame()
    validation_df = pd.DataFrame()

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folds = [[[1,2,3], 4, 5], [[2,3,4], 5, 1], [[3,4,5],1,2], [[4,5,1],2,3], [[5,1,2],3,4]]
    for fold, (trn_idx, val_idx, tst_idx) in enumerate(folds):
        if config["debug"] > 0 and fold > 0: # debug
            break

        print(f'Training with fold {fold} started (train:{trn_idx}, val:{val_idx}, test:{tst_idx})')

        # train section
        ## train dataset
        train_loader, val_loader = set_train_dataloader(df=image_df,
                                                        input_shape=config["img_size"],
                                                        train_bs=config["train_bs"],
                                                        valid_bs=config["valid_bs"],
                                                        num_workers=config["num_workers"],
                                                        trn_idx=trn_idx,
                                                        val_idx=val_idx)

        ## train model
        model = HepaClassifier(model_arch=config['model_arch'],
                               n_class=2,
                               model_shape=config["model_shape"],
                               pretrained=config["pretrained"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config['lr'],
                                    weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=config['T_0'],
                                                                        T_mult=1,
                                                                        eta_min=config['min_lr'],
                                                                        last_epoch=-1)
        er = EarlyStopping(config['patience'])
        loss_tr = torch.nn.CrossEntropyLoss().to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        print(f'Training start')

        ## training start
        for epoch in range(config['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, config['verbose_step'],scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                monitor = valid_one_epoch(epoch, model, loss_fn, val_loader, device, config['verbose_step'], scheduler=None, schd_loss_update=False)

            # Early Stopping
            if er.update(monitor[config["monitor"]], epoch, config["mode"]) < 0:
                break
            if epoch == er.val_epoch:
                torch.save(model.state_dict(),f'save/{config["model_arch"]}_{epoch}')

        del model, optimizer, train_loader, val_loader,  scheduler
        torch.cuda.empty_cache()

        # infer section
        ## infer dataset
        tst_loader, tst_df, val_loader, val_df = set_infer_dataloader(df=image_df,
                                                                      input_shape=config["img_size"],
                                                                      valid_bs=config["valid_bs"],
                                                                      num_workers=config["num_workers"],
                                                                      val_idx=val_idx,
                                                                      tst_idx=tst_idx)

        ## infer model
        model = HepaClassifier(config['model_arch'], 2, config["model_shape"]).to(device)
        model.load_state_dict(torch.load(f'save/{config["model_arch"]}_{er.val_epoch}'))

        tst_preds = []
        val_preds = []

        with torch.no_grad():
            tst_preds += [inference_one_epoch(model, tst_loader, device)]
            val_preds += [inference_one_epoch(model, val_loader, device)]

        tst_preds = np.mean(tst_preds, axis=0)
        val_preds = np.mean(val_preds, axis=0)

        del model, er
        torch.cuda.empty_cache()

        tst_df = tst_df[["img_path", "label"]]
        tst_df["result"] = tst_preds[:, 0]
        test_df = pd.concat([test_df, tst_df])

        val_df = val_df[["img_path", "label"]]
        val_df["result"] = val_preds[:, 0]
        validation_df = pd.concat([validation_df, val_df])

    # save to csv
    test_df.to_csv(f'{folder_path}/{config["model_arch"]}_preds.csv', index=False)
    validation_df.to_csv(f'{folder_path}/{config["model_arch"]}_val_preds.csv', index=False)

if __name__ == '__main__':
    main()
