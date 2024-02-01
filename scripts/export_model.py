# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import re

import torch

from tfold.config import get_config, CN
from tfold.model import PPIModel
from tfold.utils import setup


def export_ab_model():
    model_dir = '/mnt/ai4x_ceph/fandiwu/buddy1/Pre-trained.Models/tFold-Ab-models-release'
    base_model = f"{model_dir}/v8_2/model.pth"
    state = {}
    state["model"] = torch.load(base_model, map_location="cpu")
    config = get_config("tfold_ab")
    config.model.use_residue_embedding = True
    print(config)
    state["config"] = config.to_dict()
    torch.save(state, f"{model_dir}/deploy/tfold_ab_v8.pth")


def export_ag_model():
    model_dir = '/mnt/ai4x_ceph/fandiwu/buddy1/Pre-trained.Models/tFold-Ag-models-release'
    base_model = f"{model_dir}/base/model.pth"
    state = {}
    state["model"] = torch.load(base_model, map_location="cpu")
    config = get_config("tfold_ag")
    config.update(CN({
        "model": {"docking": {"use_icf": False}}
    }))
    print(config)
    state["config"] = config.to_dict()
    torch.save(state, f"{model_dir}/deploy/tfold_ag_base.pth")


def export_ppi_mdoel():
    model_dir = '/mnt/ai4x_ceph/fandiwu/buddy1/Pre-trained.Models/tFold-Ab-models-release'
    path = f"{model_dir}/esm2-650m-lr1e-05-ep100-v2/model.pth"
    state = torch.load(path, map_location="cpu")
    regex = re.compile(r'^model\.')
    state = {re.sub(regex, '', k): v for k, v in state.items()}
    config = {
        "encoder_embed_dim": 1280,
        "encoder_layers": 33,
        "encoder_attention_heads": 20,
        "token_dropout": True,
        "use_crp_embeddings": True
    }
    model = PPIModel(**config)
    model.model.load_state_dict(state)
    config["tokenizer"] = "ESM-1b"
    state = {}
    state["model"] = model.state_dict()
    state["config"] = config
    print(state["config"])
    torch.save(state, f"{model_dir}/deploy/ppi_esm_650m.pth")


def main():
    setup(True)
    export_ab_model()
    # export_ag_model()
    # export_ppi_mdoel()


if __name__ == '__main__':
    main()
