### 1. Add CIFAR10 datamodule (see how MNIST is integrated with the template, and similarly integrate CIFAR10)

Create a class in file ```src/datamodules/cifar10_datamodule.py```

Configured for instantiation ```configs/datamodule/ciafr10.yaml```

```
_target_: src.datamodules.cifar10_datamodule.CIFAR10DataModule
data_dir: ${paths.data_dir}
batch_size: 128
train_val_test_split: [45_000, 5_000, 10_000]
num_workers: 0
pin_memory: False

```

### 2. Use timm pretrained model

Create a new **LightningModule** class named **TIMMLitModule** in ```src/models/timm_module.py```

Configured for instantiation ```configs/models/timm.yaml```

```
_target_: src.models.timm_module.TIMMLitModule

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
  weight_decay: 0.0

net:
  _target_: timm.create_model
  model_name: resnet18
  pretrained: True
  num_classes: 10 

```

Create a ```confgs/experiment/cifar.yaml``` to run any experiment by overriding the model, datamodule or model params


Create a ```confgs/hparams_search/cifar10_optuna.yaml``` to tune hyperparameters

Update ```confgs/train.yaml``` and ```congs/eval.yaml``` with model config and data module config

```
defaults:
  - _self_
  - datamodule: cifar10.yaml
  - model: timm.yaml
```

### 3. Include a Makefile for building the docker image

Create a ```Dockerfile```

Update the ```Makefile```

### 4. Include scripts train.py and eval.py for training and eval(metrics) for the model, docker run <image>:<>tag python3 src/train.py experiment=experiment_name.yaml

Volume Mount  
```
docker run --volume `pwd`:/workspace/project/ pl-hydra-timm:latest python3 src/train.py experiment=cifar.yaml

```


### 5.Include COG into this for inference only (image) (optional)

Inference of any pretrained timm model

-------------------------------------------------------------------------------------------------------------------------

# DVC

[Getting started with DVC](https://dvc.org/doc/start/data-management)

Very similar to git commands to manage data verion control

### Install DVC

Installing DVC  

Ubuntu
```
sudo wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list
wget -qO - https://dvc.org/deb/iterative.asc | sudo apt-key add -
sudo apt update
sudo apt install dvc
```

OR 

```
pip install dvc
```

1. When already in git repository - ( you already will have git init, check ```ls -al``` find .git folder)
```
dvc init
```

Now, there will be ```.dvc``` folder created

2. Add **data** folder to track
```
dvc add data
```
A new file ```data.dvc``` md5 hash created to track all changes

3. **autostage**: if enabled, DVC will automatically stage (git add) DVC files created or modified by DVC commands.
```
dvc config core.autostage true
```

4. **Check remote** where we can push ( this is for git, we need same for dvc)
```
git remote -v
```

5. Go and create a new folder ( now in gdrive ), say lightning-hydra

get into the folder and check the url as below - 

https://drive.google.com/drive/u/1/folders/1t9Vs8OwPOtQGnz1aR4KyPQA2k7FbKR5A

folder id - 1t9Vs8OwPOtQGnz1aR4KyPQA2k7FbKR5A

Add a remote

```
dvc remote add gdrive gdrive://1t9Vs8OwPOtQGnz1aR4KyPQA2k7FbKR5A
```

6. Add folders and files to stage
```
git add .
```

```
git commit -m "updated dvc"
```

7. Push the changes to gdrive ( NOTE : give permission to folders and gmail account)

-r : remote, gdrive : name of remote folder

```
dvc push -r gdrive
```

------------------------------------------------------------------------------------------------

## DVC Pipelines (DAG)

Read/ Watch [here](https://dvc.org/doc/start/data-management/data-pipelines)

1. Need to have a ```dvc.yaml``` describing stages and dependecies, see below -
```
stages:
  train-mnist:
    cmd: python3 src/train.py experiment=mnist
    deps:
      - data/MNIST
```

2. To run pipelines, dvc reproduce
```
dvc repro train-mnist
```

3. ...


--------------------------------------------------------------------------------------


### NOTE

✨💡✨ One folder or file cannot be tracked by both - git or dvc. 

To fix for gitpod.io

ssh -L 8080:localhost:8080 <ssh string from gitpod>

To see all files/folders, size, users and permisions
```
ls -alrth
```