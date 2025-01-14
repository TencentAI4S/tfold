# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import re

import torch

from tfold.config import get_config, CN
from tfold.model import PPIModel, ComplexStructureModel, ComplexLiteModel
from tfold.utils import setup


def export_ppi_mdoel(path, ouput):
    state = torch.load(path, map_location="cpu")
    regex = re.compile(r'^model\.')
    state = {re.sub(regex, '', k): v for k, v in state.items()}
    config = get_config("esm", clone=True)
    model = PPIModel(config)
    model.model.load_state_dict(state)
    state = {}
    state["model"] = model.state_dict()
    state["config"] = config.to_dict()
    print(state["config"])
    torch.save(state, ouput)


def export_complex_model(path, output):
    state = {}
    state["model"] = torch.load(path, map_location="cpu")
    config = get_config("tfold_ab")
    print(config)
    state["config"] = config.to_dict()
    # check loading success
    model = ComplexStructureModel(config)
    model.load_state_dict(state["model"])
    torch.save(state, output)


def export_ab_model():
    model_dir = "/mnt/fandiwu/buddy1/Pre-trained.Models/tFold-Ab-models-release"
    save_dir = "../checkpoints"
    raw_ppi_path = f"{model_dir}/esm2-650m-lr1e-05-ep100-v2/model.pth"
    ppi_path = f"{save_dir}/esm_ppi_650m_ab.pth"
    export_ppi_mdoel(raw_ppi_path, ppi_path)

    trunk_model = f"{save_dir}/tfold_ab_trunk_v8.2.pth"
    export_complex_model(f"{model_dir}/v8_2/model.pth", trunk_model)

    ppi_state = torch.load(ppi_path, map_location='cpu')
    cfg = get_config()
    cfg.ppi = CN(ppi_state['config'])
    trunk_path = f"{save_dir}/tfold_ab_trunk.pth"
    trunk_state = torch.load(trunk_path, map_location='cpu')
    cfg.trunk = CN(trunk_state['config'])
    model = ComplexLiteModel(cfg)
    model.ppi.load_state_dict(ppi_state['model'])
    model.trunk.load_state_dict(trunk_state['model'])
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict()
    }, f"{save_dir}/tfold_ab.pth")


def export_ag_model():
    model_dir = '/mnt/fandiwu/buddy1/Pre-trained.Models/tFold-Ag-models-release'
    save_dir = "../checkpoints"
    base_model = f"{model_dir}/base/model.pth"
    state = {}
    state["model"] = torch.load(base_model, map_location="cpu")
    config = get_config("tfold_ag")
    config.update(CN({
        "model": {"docking": {"use_icf": False}}
    }))
    print(config)
    state["config"] = config.to_dict()
    torch.save(state, f"{save_dir}/tfold_ag_base.pth")

    base_model = f"{model_dir}/ppi/model.pth"
    config = config.clone()
    config.model.docking.use_icf = True
    print(config)
    state = {}
    state["model"] = torch.load(base_model, map_location="cpu")
    state["config"] = config.to_dict()
    torch.save(state, f"{save_dir}/tfold_ag_ppi.pth")


def export_tcr_model():
    model_dir = '/mnt/fandiwu/buddy1/Pre-trained.Models/tFold-TCR-models-V2.2-release'
    save_dir = "../checkpoints"
    export_ppi_mdoel(f"{model_dir}/ESM-PPI-tcr/model.pth", f"{save_dir}/esm_ppi_650m_tcr.pth")
    base_model = f"{model_dir}/pMHC/model.pth"
    state = {}
    state["model"] = torch.load(base_model, map_location="cpu")
    config = get_config("tfold_ab", clone=True)
    config.model.use_residue_embedding = False
    print(config)
    state["config"] = config.to_dict()
    # check loading success
    model = ComplexStructureModel(config)
    model.load_state_dict(state["model"])
    torch.save(state, f"{save_dir}/tfold_pmhc_trunk.pth")

    base_model = f"{model_dir}/TCR/model.pth"
    state = {}
    state["model"] = torch.load(base_model, map_location="cpu")
    config = get_config("tfold_ab", clone=True)
    config.model.use_residue_embedding = False
    print(config)
    state["config"] = config.to_dict()
    torch.save(state, f"{save_dir}/tfold_tcr_trunk.pth")

    base_model = f"{model_dir}/Complex/model.pth"
    state = {}
    state["model"] = torch.load(base_model, map_location="cpu")
    config = get_config("tfold_tcr", clone=True)
    print(config)
    state["config"] = config.to_dict()
    torch.save(state, f"{save_dir}/tfold_tcr_pmhc_trunk.pth")


def main():
    setup(True)
    # export_ab_trunk_model()
    # export_ab_model()
    # export_ag_model()
    export_tcr_model()


if __name__ == '__main__':
    main()
