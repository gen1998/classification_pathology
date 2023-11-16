# Running thecode
## Setup Environment
1. clone [timm](https://github.com/huggingface/pytorch-image-models#getting-started-documentation)
2. create folder that is saving model weights.
3. create and activate anaconda environment
```
git clone git@github.com:huggingface/pytorch-image-models.git
mkdir save
conda env create -f conda_env/environment.yml
conda activate environment
```

## Setup Your Dataset
1. Put all your dataset images together in the same folder and rewrite `BASE_TRAIN_IMAGE_PATH` in `./config/setting.py` as the folder path.
2. create the .csv that summarizes the information on the images for the training as shown in `./data/input/input.csv`.

   info :
   ```
   columns  : img_name, label, fold
   img_name : all(train, validation, test..) image
   label    : label for each img_name, labels must be intengers and start from 0 (0, 1, 2..).
   fold     : fold(5CV) for each img_name, labels must be 0, 1, 2, 3, or 4.
   (Although fold can be specified in the code, it should be set in advance because mdedical images are subject to various biases depending on the data.
   If you are interested, check Stratified k-Fold Cross-Validation or patient stratification cross-validation)
   ```



# TODO
- use various folds.
- 
