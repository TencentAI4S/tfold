# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import os

import torch
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_

from tfold.utils import setup


def main():
    setup(True)
    model_dir = "/mnt/fandiwu/buddy1/Pre-trained.Models/tFold-Ag-models-release"
    path = f"{model_dir}/psp/params_model_4_ptm.npz"
    model_version = "_".join(os.path.basename(path).split("_")[1:]).split('.')[0]
    config = model_config(model_version)
    config.data.common.max_recycling_iters = 0
    config.data.predict.max_msa_clusters = 384
    config.globals.use_lma = False
    config.globals.offload_inference = False
    model = AlphaFold(config)
    model = model.eval()
    import_jax_weights_(model, path, version=model_version)
    state = {
        "model": model.state_dict(),
        "config": config.to_dict()
    }
    torch.save(state, "../checkpoints/alphafold_4_ptm.pth")


if __name__ == '__main__':
    main()
