# Running thecode
## Setup
### Setup Environment
1. clone [timm](https://github.com/huggingface/pytorch-image-models#getting-started-documentation)
2. create folder that is saving model weights.
3. create and activate anaconda environment
```
git clone git@github.com:huggingface/pytorch-image-models.git
mkdir save
conda env create -f conda_env/environment.yml
conda activate environment
```

### Setup Your Dataset
1. Put all your dataset images together in the same folder and rewrite `BASE_TRAIN_IMAGE_PATH` in `./config/setting.py` as the folder path.
2. create the .csv that summarizes the information on the images for the training as shown in `./data/input/input.csv`.

   CSV Info :
   ```
   columns  : img_name, label, fold
   img_name : all(train, validation, test..) image
   label    : label for each img_name, labels must be intengers and start from 0 (0, 1, 2..).
   fold     : fold(5CV) for each img_name, labels must be 0, 1, 2, 3, or 4.
               (Although fold can be specified in the code, it should be set in advance because mdedical images are subject to various biases depending on the data.
               If you are interested, check Stratified k-Fold Cross-Validation or patient stratification cross-validation)
   ```

### Setup Parameter
Basically, you can set new parameters by rwriting `.config/default.json`

Parameters :
```
seed : random seed [default: 6]
debug : to debug [default: False]
device : Context-manager that changes the selected device [default: "cuda:0"]
folder_name : folder name saved infer result [default: ""]

model_arch : timm model name (timm.list_models()) [default: "tf_efficientnet_b3"]
model_shape : backbone model name. Resnet:"res" / Efficientnet:"eff" / Vision Transfomer:"vit". [default: "eff"]
pretrained : pretrained model [default: True]

img_size : input img pixels [default: 256]
train_bs : training data batch size [default: 128]
valid_bs : validation and test batch size [default: 128]
epochs : max epochs [default: 100]
verbose_step :
num_workers : 

Optimizer Parameters
lr : learning rate [default: 3e-4]
weight_decay : weight decay (L2 penalty) [default: 1e-4]
T_0 : Number of iterations for the first restart [default: 10]
min_lr : Minimum learning rate [default: 3e-6]
If you are interested, you check "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts".

EarlyStopping Parameters(Stop training when a monitored metric has stopped improving. To prevent overfitting)
monitor : monitor variables ("val_loss" or "val_accuracy") [default: "val_accuracy"]
patience : epochs to wait not to imporve [default: 2]
mode : which direction of improvement ("val_loss":"min" / "val_accuracy":"max") [default: "max"]
```

## Train and infer
```
conda activate environment.py
python run.py
```


# TODO
- use various folds.
- 
