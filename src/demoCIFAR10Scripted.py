import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig
from torchvision import transforms
import tarfile

from src import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    #my_tar = tarfile.open('model.tar.gz')
    #print(my_tar)
    #my_tar.extractall()
    #print('--------**---------')
    #my_tar.close()

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")
    
    model.eval()

    # Labels for CIFAR10 dataset
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 

    def classify_top10(inp):
        inp = transforms.ToTensor()(inp).unsqueeze(0)
        with torch.no_grad():
            prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
            confidences = {labels[i]: float(prediction[i]) for i in range(10)}    
        return confidences
    
    im = gr.Image(shape=(224, 224), type="pil")

    demo = gr.Interface(
        fn=classify_top10,
        inputs=[im],
        outputs=gr.Label(num_top_classes=10),
    )

    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demoMnistScripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()