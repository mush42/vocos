# coding: utf-8

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn


from vocos.pretrained import Vocos


DEFAULT_OPSET_VERSION = 15
_LOGGER = logging.getLogger("export_onnx")


class VocosGen(nn.Module):
    def __init__(self, vocos):
        super().__init__()
        self.vocos = vocos

    def forward(self, mels):
        x = self.vocos.backbone(mels)
        audio_output = self.vocos.head(x)
        return audio_output


def export_generator(config_path, checkpoint_path, output_dir, opset_version):
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    class_module, class_name = config["model"]["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    vocos_cls = getattr(module, class_name)

    components = Vocos.from_hparams(config_path)
    params = config["model"]["init_args"]
    
    vocos = vocos_cls.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        feature_extractor=components.feature_extractor,
        backbone=components.backbone,
        head=components.head,
        sample_rate=params["sample_rate"],
        initial_learning_rate=params["initial_learning_rate"],
        num_warmup_steps=params["num_warmup_steps"],
        mel_loss_coeff=params["mel_loss_coeff"],
        mrd_loss_coeff=params["mrd_loss_coeff"],
    )

    model = VocosGen(vocos)
    model.eval()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    epoch = state_dict["epoch"]
    global_step = state_dict["global_step"]
    onnx_filename = f"vocos-epoch={epoch}.step={global_step}.onnx"
    onnx_path = output_dir.joinpath(onnx_filename)

    dummy_input = torch.rand(1, vocos.backbone.input_channels, 64)
    dynamic_axes = {
        "mels": {0: "batch_size", 2: "time"},
        "audio": {0: "batch_size", 1: "time"},
    }

    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=os.fspath(onnx_path),
        input_names=["mels"],
        output_names=["audio"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True,
    )
    return str(onnx_path)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        prog="export_onnx",
        description="Export a model checkpoint to onnx",
    )

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET_VERSION)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _LOGGER.info("Exporting model to ONNX")
    _LOGGER.info(f"Config path: `{args.config}`")
    _LOGGER.info(f"Using checkpoint: `{args.checkpoint}`")
    onnx_path = export_generator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        opset_version=args.opset
    )
    _LOGGER.info(f"Exported ONNX model to: `{onnx_path}`")


if __name__ == '__main__':
    main()