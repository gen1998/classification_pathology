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
1. create the .csv that summarizes the information on the images for the training as shown in `./data/input/input.csv`
