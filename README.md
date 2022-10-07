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
```

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
```
def validation_epoch_end(self, outputs: List[Any]):
      acc = self.val_acc.compute() # get current val acc
      self.val_acc_best(acc) # update best so far val acc

     # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
     # otherwise metric would be reset by lightning after each epoch
     self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

     loss = self.val_loss.compute() #add validation loss to hp_metric
     self.log("hp_metric", loss)
```

## 3. Do a Hyperparam sweep for CIFAR10 dataset with resnet18 from timm.
### Define hyperparameter search space
```
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      model.optimizer._target_: choice(torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop)
      datamodule.batch_size: choice(64, 128, 256)
```

### Find the best batch_size and learning rate, and optimizer ( under limited runs on colab gpu)
```
Best Params :

              Batch size : 64

              Learning rate : 0.068378

              HP_Metric/ Validation Loss : 0.19418

              Optimizer : SGD
```

### Link to public drive folder dvc :

https://drive.google.com/drive/folders/1t9Vs8OwPOtQGnz1aR4KyPQA2k7FbKR5A?usp=sharing 
 

### Link to Tensorboard :

https://tensorboard.dev/experiment/d93drFa9QbaCP4MFK4tv4A/#scalars


✨💡✨
```- Overrides the model.yaml``` settings in model folder

------------------------------

# Deployment for Demos

[Checkout Gradio demo](https://github.com/aiplaybookin/gradio-demo) to get some flavour.

------------------------------

# Add Demo App

Add mnistDemo.py

Script is similar to ```train.py``` or ```eval.py``` structure wise.

Import necessary libraries

Define a function demo which would be called in main with configurations provided by config yaml () -
a. Checks whether model (check point) path is given or not
b. Instansiate model for inference (.pth file)
c. Loads weights, model
d. Define interface function

```source = "canvas"``` : User can draw and we infer
```image mode = "L" ```: Single channel (because we using MNIST dataset which is single channel -B&W)
```invert color = true ```- Because when we use canvas digits are black in color and background in white
```live = true```: Realtime inferenece applications 

Add gradio in requirements.txt
```
gradio==3.3.1
```

Create a config file ```demoMnist.yaml``` to fetch configurations while running app under config folder ( this is similar to eval.yaml file)


Still needs :
```
- callbacks: default.yaml
- experiment: null
```

To assert : make it mandatory to provide ckpt path use as below -
```
ckpt_path: ???
```

Run
```
python src/demoMnist.py ckpt_path=logs/train/runs/2022-09-30_06-02-58/checkpoints/last.ckpt experiment=mnist
```
-------------------

### **Drawbacks :**
We need to provide data module, experiment, trainer, parameters to instansiate the model everytime

Create model - load weights, having model.py (e.g. mnist_module.py)

-------------------
# TorchScript

TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.

TorchScript is a statically typed subset of Python that can either be written directly (using the [TORCH.JIT.SCRIPT](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script) decorator) or generated automatically from Python code via **Tracing**. When using tracing, code is automatically converted into this subset of Python by recording only the actual operators on tensors and simply executing and discarding the other surrounding Python code.

In other words -
***You can export as non python representaion of the model to be loaded by any environment ( e.g. pure c++ )***

Provides Efficient and **Portable** Pytorch production deployment 

[Read more here](https://paulbridger.com/posts/mastering-torchscript/)

[Baics Tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)


## Script vs Tracing

```torch.jit.script``` captures both the operations and full conditional logic of your model, whereas ```torch.jit.trace``` will actually run the model with given dummy inputs and it will freeze the conditional logic as per the dummy values provided.


You can read about edge cases of Tracing [here:](https://pytorch.org/docs/master/jit.html#tracing-edge-cases)

[Compile your model to TorchSript example](https://pytorch-lightning.readthedocs.io/en/stable/deploy/production_advanced_2.html)


```LightningModule``` has a handy method ```to_torchscript()``` that returns a scripted module which you can save or directly use


💡If you want to script a different method (export a function with torch script or trace), you can decorate the method with ```torch.jit.export()```



### Modifications for Advanced Deployments (Torch Script, demoMnistScripted.py):

1. Imports in mnist_module.py file to export the tranformations also
```
import torch.nn.functional as F
from torchvision import transforms as T
```

Add below lines above forward function in mnist_module.py.
```
self.predict_transform = T.Normalize((0.1307,), (0.3081,))      
```

Add below line jst after the forward function, this exports the forward_jit func whenver there is torchscript or trace. Here we are push the necessary transforms also to model instansiation. So that nothing is required.
```
@torch.jit.export                     
    def forward_jit(self, x: torch.Tensor):
        with torch.no_grad():
            # transform the inputs
            x = self.predict_transform(x)

            # forward pass
            logits = self(x)

            preds = F.softmax(logits, dim=-1)

        return preds
```

2. In train.py to save the serialized model (or complied model). Add below line just after - train_metrics = trainer.callback_metrics
```
log.info("Scripting Model..")

    scripted_model = model.to_torchscript(method="script")
    torch.jit.save(scripted_model, f"{cfg.paths.output_dir}/model.script.pt")

    log.info(f"Saving traced model to {cfg.paths.output_dir}/model.script.pt")
```


3. demoMnistScripted.yaml

```
# @package _global_

defaults:
  - _self_
  - paths: default.yaml
  - hydra: default.yaml

task_name: "demo_traced"

# checkpoint is necessary for demo
ckpt_path: ???
```


Run
```
python3 src/demoMnistScripted.py ckpt_path=logs/train/runs/2022-09-30_12-43-23/model.script.pt
```
-------------------

## Dockerize the demoMnistScipted

Create a Dockerfile named ```Dockerfile.demoMNIST``` 
```
contents
```

Copy deployable model to root folder ( or some deploy folder)
```
cp logs/train/runs/2022-09-30_12-43-23/model.script.pt model.script.pt
```

Create / add files and folders not required in ```.dockerignore``` : To reduce the docker image size
```
logs/
data/
tests/
```



Docker build by  -
```
docker build -f Dockerfile.demoMNIST -t testapp .
```

Docker run by
```
docker run -it -p 8080:8080 testapp:latest
```

-------------------
# 4. Deployment for CIPFAR10 Model (using Torch Script)
-------------------

## Update the scripts

1. In train.py to save the serialized model (or complied model). Add below line just after - train_metrics = trainer.callback_metrics
```
log.info("Scripting Model..")

    scripted_model = model.to_torchscript(method="script")
    torch.jit.save(scripted_model, f"{cfg.paths.output_dir}/model.script.pt")

    log.info(f"Saving traced model to {cfg.paths.output_dir}/model.script.pt")
```

2. Create a new file ```demoCIFAR0Scripted.yaml``` inside config folder containing following lines of code. It helps set default paths and makes it mandatory to provide model path to load.

```
# @package _global_

defaults:
  - _self_
  - paths: default.yaml
  - hydra: default.yaml

task_name: "demo_traced"

# checkpoint is necessary for demo
ckpt_path: ???
```

3. Train for few epochs ( I did only for on epoch)
```
python3 src/train.py experiment=cifar
```

4. Create a new file ```demoCIFAR10Scripted.py``` under ```src``` folder 

- Create a Gradio app interface
- Loads the model
- It must accept image from user, and give the top 10 predictions

```
import pyrootutils

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
from torchvision import transforms

from src import utils
```
5. test 
```
python3 src/demoCIFAR10Scripted.py ckpt_path=logs/train/runs/2022-10-05_05-12-27/model.script.pt
```

## Dockerize and Push to Dockerhub

1. Create a dockerfile named ```Dockerfile.demoCIFAR10```

2. Create a requirements file named ```requirements_cpu.txt```

3. Docker build by  -
```
docker build -f Dockerfile.demoCIFAR10 -t vikashkr117/gradio-cifar-app .
```

4. Docker run by
```
docker run -it -p 8080:8080 vikashkr117/gradio-cifar-app:latest
```

5. Login to docker hub from CLI
```
docker login -u vikashkr117
```
At the password prompt, enter the personal access token.

6. Push the image 
```
docker push vikashkr117/gradio-cifar-app
```

## Instructions to use app 
### (via docker image)

1. Go to [Play with Docker](https://labs.play-with-docker.com) and start a session. Pull the image using below command
```
docker pull vikashkr117/gradio-cifar-app
```

2. Run the pulled docker image
```
docker run -it -p 8080:8080 vikashkr117/gradio-cifar-app:latest
```

3. Copy the gradio link provide e.g. https://<......>.gradio.app and paste in web browser

4. Place an image and click on submit, you will see top 10 classes and probabilities associated with it.

5. Cheers 🥂
-------------------

python3 src/demoCIFAR10Scripted.py ckpt_path=logs/train/runs/2022-10-05_05-12-27/model.script.pt

Ref : https://gradio.app/image_classification_in_pytorch/

# NOTES :

✨💡✨ One folder or file cannot be tracked by both - git or dvc- Yes/No?

To fix for gitpod.io (or just use gitpod in local VS code)

ssh -L 8080:localhost:8080 <ssh string from gitpod>

To see all files/folders, size, users and permisions  
```
ls -alrth
```


Run the tests locally using ```pre-commit run —all-files```

This will run all the tests defined in the ```.pre-commit-config.yaml``` file

To look into docker files :-
```
docker container run --rm -it testapp /bin/sh
```


⚠️ You should not start a container just to see the image contents. For instance, you might want to look for malicious content, not run it. Use "create" instead of "run";

```
docker create --name="tmp_$$" image:tag
docker export tmp_$$ | tar t
docker rm tmp_$$
```

Dockerfile ignore 
```
# 1. Ignore everything
**

# 2. Add files and directories that should be included
!configs/**
!src/**
!package.json
!package-lock.json

# 3. Bonus step: ignore any unnecessary files that may be inside those allowed directories in 2
**/*~
**/*.log
**/.DS_Store
**/Thumbs.db
```

docker image history

RUN apt install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists

# tell the port number the container should expose
EXPOSE 8080

# run the application
CMD [ "python3", "src/demoMnistScripted.py" , "ckpt_path=model.script.pt"]

1. Create a tar file 

tar -czvf model.tar.gz demo_model