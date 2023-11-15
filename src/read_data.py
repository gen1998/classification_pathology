import random
import os
import pandas as pd

from config.settings import BASE_TRAIN_IMAGE_PATH

def data_load() -> pd.DataFrame:
    train_df = pd.read_csv(f"./data/input/input.csv")
    train_df["img_path"] = train_df.img_name.apply(lambda x:BASE_TRAIN_IMAGE_PATH+x)
    train_df = train_df.reset_index(drop=True)

    return train_df
