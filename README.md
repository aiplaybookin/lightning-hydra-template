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

