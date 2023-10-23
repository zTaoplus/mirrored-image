import argparse
from typing import *

import kserve
import torch

from transformers import pipeline


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

DEFAULT_MODEL_PATH = "/mnt/models"


class PromptInjection(kserve.Model):
    def __init__(self, name: str, protocol: str, model_path: str = DEFAULT_MODEL_PATH):
        super().__init__(name)
        self.protocol = protocol
        self.ready = False
        self.load(model_path)

    def load(self, model_path: str):
        self.pipe = pipeline("text-classification", model=model_path, device=DEVICE)

        self.ready = True

    def predict(self, payload: Dict):
        """
        input:
        {
        "instances":[]
        }
        """

        inputs: list = payload["instances"]

        return {"predictions": self.pipe(inputs)}


parser = argparse.ArgumentParser(prog="prompt injection service")

# transformer's args
parser.add_argument(
    "--model_name",
    required=True,
    default="prompt_judgement",
    help="model name for serving",
)

parser.add_argument(
    "--model_path",
    default=DEFAULT_MODEL_PATH,
    help="model storage path",
)

parser.add_argument("--protocol", help="The protocol for the predictor", default="v1")

args, _ = parser.parse_known_args()


if __name__ == "__main__":
    model = PromptInjection(
        args.model_name, protocol=args.protocol, model_path=args.model_path
    )
    if not model.ready:
        raise RuntimeError("Model Cannot Load!!!")

    kserve.ModelServer().start(models=[model])
