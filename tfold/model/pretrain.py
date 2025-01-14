# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import hashlib
import os
import urllib

import torch


def alpha_fold_4_ptm():
    print("Loading AlphaFold 4 PTM model")
    return load_model_hub("alphafold_4_ptm", 'd62419bfdceaad277d14529e7bc85bb7')


def esm_ppi_650m_ab():
    print("Loading ESM PPI 650m AB model")
    return load_model_hub("esm_ppi_650m_ab", '9f332b21296d8182c6159ba7833d3a74')


def esm_ppi_650m_tcr():
    print("Loading ESM PPI 650m TCR model")
    return load_model_hub("esm_ppi_650m_tcr", 'f3827829fb45fd222cfb69d4164db9dd')


def tfold_ab_trunk():
    print("Loading tFold AB Trunk model")
    return load_model_hub("tfold_ab_trunk", '73d4f4ce4a6ff5f712bc6d89d7f3eb08')


def tfold_ab():
    print("Loading tFold AB model")
    return load_model_hub("tfold_ab", '3c240f39a00ce28982d9f6ce390a7e2a')


def tfold_ag_base():
    print("Loading tFold AG Base model")
    return load_model_hub("tfold_ag_base", 'c43e42ae294389540147bfe8a37cb5d5')


def tfold_ag_ppi():
    print("Loading tFold AG PPI model")
    return load_model_hub("tfold_ag_ppi", 'ff749ec79b2b0314f69bfb5bef2e72f8')


def tfold_pmhc_trunk():
    print("Loading tFold pMHC Trunk model")
    return load_model_hub("tfold_pmhc_trunk", 'dfae4da76fa1842af04c1211c4dd38b2')


def tfold_tcr_pmhc_trunk():
    print("Loading tFold TCR pMHC Trunk model")
    return load_model_hub("tfold_tcr_pmhc_trunk", '123f668c8b170270d86944fcae3ef65f')


def tfold_tcr_trunk():
    print("Loading tFold TCR Trunk model")
    return load_model_hub("tfold_tcr_trunk", '0e42dec469313076b706647cf30075d5')


def load_model_hub(model_name, expected_md5):
    model_path = _download_model_data(model_name, expected_md5)
    return model_path


def _download_model_data(model_name, expected_md5):
    url = f"https://zenodo.org/records/12602915/files/{model_name}.pth?download=1"
    model_path = load_hub_workaround(url, model_name)

    # Check MD5
    if not check_md5(model_path, expected_md5):
        raise ValueError(f"MD5 mismatch for {model_name}. Please try downloading the model again.")

    return model_path


def load_hub_workaround(url, model_name):
    try:
        os.makedirs(f"{torch.hub.get_dir()}/checkpoints", exist_ok=True)
        model_path = f"{torch.hub.get_dir()}/checkpoints/{model_name}.pth"
        if os.path.exists(model_path):
            return model_path
        torch.hub.download_url_to_file(url, progress=True, dst=model_path)
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url} with {e}, check if you specified a correct model name.")
    return model_path


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_md5(file_path, expected_md5):
    return calculate_md5(file_path) == expected_md5
