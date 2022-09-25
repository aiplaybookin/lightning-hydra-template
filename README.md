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

4. **Check remote** where we can push/push ( this is for git, we need same for dvc)
```
git remote -v
```

5. Go and create a new folder ( now in gdrive ), say lightning-hydra

get into the folder and check the url as below - 
https://drive.google.com/drive/u/1/folders/1t9Vs8OwPOtQGnz1aR4KyPQA2k7FbKR5A
https://drive.google.com/drive/u/1/folders/1ts8OwPOtQGnz1aR4KyPQA2k7FbKR5A

folder id - 1ts8OwPOtQGnz1aR4KyPQA2k7FbKR5A

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
## Hyperparameters Tuning

### Optuna

1. lets just run for normal grid search before using optuna

```-m`` because we need multiple runs with different batch
This will run 4 jobs in parallel

```
python3 src/train.py -m experiment=mnist datamodule.batch_size=16,32,64,128 tags=["batch_size_exp"]
```

💡✨ **Median Pruner** : Prune if the trial’s best intermediate result is worse than median of intermediate results of previous trials at the same step.

--------

Pure Optuna example code [here](https://colab.research.google.com/drive/13PFJfRTYR-B_DbzrWdp8iiGF7yUO0qcO?usp=sharing)

**Hydra plugin** for Optuna [here](https://hydra.cc/docs/plugins/optuna_sweeper/)

Setup hydra yaml file under ```configs/hparams_search/mnist_optuna.yaml```

1. Choose metric to be optimised : this should be logged in ```training_step``` of ```model class``` in lightning module! here **MNISTLitModule**
2. Mention direction : maximise or minimise according to metric used
3. Total number of trails to run
4. Choice of Sampler : TPE is bayesian
5. n_startup_trials: 10 # number of random sampling runs before optimization starts
6. Define hyperparameter search space
```
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      datamodule.batch_size: choice(32, 64, 128, 256)
      model.net.lin1_size: choice(64, 128, 256)
      model.net.lin2_size: choice(64, 128, 256)
      model.net.lin3_size: choice(32, 64, 128, 256)
```

To run the hyperparameter search 

```
python train.py -m hparams_search=mnist_optuna experiment=mnist
```

-----------------------

## 🪵 Loggers in Pytorch Lightning

Read [here](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html)

1. CometLogger - Track your parameters, metrics, source code and more using Comet.

2. CSVLogger - Log to local file system in yaml and CSV format.

3. MLFlowLogger - Log using MLflow.

4. NeptuneLogger - Log using Neptune.

5. TensorBoardLogger - Log to local file system in TensorBoard format.

6. WandbLogger - Log using Weights and Biases.

Remote Logging with PyTorch Lightning: https://pytorch-lightning.readthedocs.io/en/stable/common/remote_fs.html (Links to an external site.)


### To Enable loggers -

```train.yaml``` had logger specified as null (default).

To enable logger, we should add below line to the yaml file specific to experiment in experiment folder (for e.g. ```mnist.yaml``` in experiment folder) to use particular say, tensorboard logger.
```
- override /logger: tensorboard.yaml
```

Run and check
```
python3 src/train.py experiment=mnist
````

Go to tensorboard folder within logs folder (bind_all : someone else can access in same network)
```
tensorboard --logdir . --bind_all
```
It will open in local browser..

NOTE : We can change in train.yaml also which will become a default for all whether you run with experiment or without

To Log to multiple logger we have ```many_logger.yaml``` in logger folder. This contains list of loggers say - tensorboard, csvlogger, mlflow etc. We could 
```
- override /logger: many_loggers.yaml
```

Say ```MLFLOW``` is also enabled via ```many_loggers.yaml``` file

Run the experiment again ```python3 src/train.py experiment=mnist```

Go to the mlflow folder inside the logs (must have child folder created - ./mlruns/mlruns/meta.yaml) and run below command.
You will be able to see the logs for hyperparams, accuracy etc.
```
mlflow ui
```

------------

## 1. Add step wise logging
In training class, in training step, while saving logs set ```on_step=True``` as shown below -

```
def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
```

## 2. log hp_metric in tensorboard as validation loss

## 3. Do a Hyperparam sweep for CIFAR10 dataset with resnet18 from timm.

### Find the best batch_size and learning rate, and optimizer

```
    # define hyperparameter search space
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      model.optimizer._target_: choice(torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop)
      datamodule.batch_size: choice(64, 128, 256)
```

- Overrides the model.yaml settings in model folder

------------

# NOTES :

✨💡✨ One folder or file cannot be tracked by both - git or dvc- Yes/No?

To fix for gitpod.io

ssh -L 8080:localhost:8080 <ssh string from gitpod>

To see all files/folders, size, users and permisions
```
ls -alrth
```


Run the tests locally using ```pre-commit run —all-files```

This will run all the tests defined in the ```.pre-commit-config.yaml``` file