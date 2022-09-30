import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Tuple

import torch
import hydra
import gradio as gr 
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torchvision.transforms as T
import torch.nn.functional as F

from src import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    ckpt = torch.load(cfg.ckpt_path)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    log.info(f"Loaded Model: {model}")

    transforms = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def recognize_digit(image):
        if image is None:
            return None
        image = transforms(image).unsqueeze(0)
        logits = model(image)
        preds = F.softmax(logits, dim=1).squeeze(0).tolist()
        return {str(i): preds[i] for i in range(10)}

    im = gr.Image(shape=(28, 28), image_mode="L", invert_colors=True, source="canvas")

    demo = gr.Interface(
        fn=recognize_digit,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch()

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="demoMnist.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()