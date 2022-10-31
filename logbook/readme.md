<table align="center"><tr><td align="center" width="9999">

# Distributed Training


</td></tr></table>

### To do
Model: timm.create_model("vit_base_patch32_224", pretrained=True)

Dataset: CIFAR10

Epochs: >25

1. Train ViT Base using **FSDP (4 GPU)**
2. Train ViT Base using **DDP (4 GPU x 2 Nodes)**
3. Use the highest batch_size possible for both strategies
4. Store the best checkpoint of both to AWS S3
5. In your repository add a training log book folder and create a .md file for above experiments
6. Add Model Validation Metrics, Training Time, GPU memory usage, GPU Utilization (nvidia-smi dump) for both strategies
 - you can run the testing script after training the model to get the final model metrics
7. Add the maximum batch_size number you were able to achieve
8. Upload Tensorboard logs to Tensorboard.dev and add link to it


# IN Progress 🙏🏽

<img src="outputs/ec2_cpu.png" align="center" width="550" >