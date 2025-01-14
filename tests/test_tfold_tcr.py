# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/14 19:13
import os
import sys

import torch
import tqdm

sys.path.append('.')

from tfold.deploy import TCRpMHCPredictor, TCRPredictor, PeptideMHCPredictor
from tfold.utils import jload
from tfold.protein.parser import parse_fasta
from tfold.model.pretrain import esm_ppi_650m_tcr, tfold_tcr_pmhc_trunk


def test_tcr_pmhc_example():
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    tasks = jload("./examples/tcr_pmhc_example.json")
    output = "./examples/results"
    batches = []
    for task in tasks:
        name = task["name"]
        task["output"] = f"{output}/{name}_test_json.pdb"
        batches.append(task)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # antibody & antigen structures prediction & sequence design
    predictor = TCRpMHCPredictor(
        # ppi_path=esm_ppi_650m_tcr(),
        # trunk_path=tfold_tcr_pmhc_trunk()
        ppi_path="checkpoints/esm_ppi_650m_tcr.pth",
        trunk_path="checkpoints/tfold_tcr_pmhc_trunk.pth"
    )
    predictor = predictor.to(device)
    if torch.cuda.is_bf16_supported():
        print("convert model to bfloat16")
        predictor.to(torch.bfloat16)

    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"], chunk_size=512)


def test_tcr_example():
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    tasks = jload("./examples/tcr_example.json")
    output = "./examples/results"
    batches = []
    for task in tasks:
        name = task["name"]
        task["output"] = f"{output}/{name}_test_tcr.pdb"
        batches.append(task)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # antibody & antigen structures prediction & sequence design
    predictor = TCRPredictor.restore_from_module(
        # ppi_path=esm_ppi_650m_tcr(),
        # trunk_path=tfold_tcr_trunk(),
        ppi_path="checkpoints/esm_ppi_650m_tcr.pth",
        trunk_path="checkpoints/tfold_tcr_trunk.pth"
    )
    predictor = predictor.to(device)
    if torch.cuda.is_bf16_supported():
        predictor.to(torch.bfloat16)

    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"])


def test_pmhc_example():
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    tasks = jload("./examples/pmhc_example.json")
    output = "./examples/results"
    batches = []
    for task in tasks:
        name = task["name"]
        task["output"] = f"{output}/{name}_test_pmhc.pdb"
        batches.append(task)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # antibody & antigen structures prediction & sequence design
    predictor = PeptideMHCPredictor.restore_from_module(
        # ppi_path=esm_ppi_650m_tcr(),
        # trunk_path=tfold_pmhc_trunk()
        ppi_path="checkpoints/esm_ppi_650m_tcr.pth",
        trunk_path="checkpoints/tfold_pmhc_trunk.pth"
    )
    predictor = predictor.to(device)
    if torch.cuda.is_bf16_supported():
        predictor.to(torch.bfloat16)

    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"])


def test_fasta():
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    fasta_path = "./examples/fasta.files/6zkw_E_D_A_B_C.fasta"

    output = "./examples/results/6zkw_E_D_A_B_C_test_fasta.pdb"
    sequences, ids, _ = parse_fasta(fasta_path)
    assert len(sequences) in (1, 2, 3, 4, 5), f"must be 2 chains in fasta file"
    chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids)]
    name = os.path.basename(fasta_path).split(".")[0]
    batches = [
        {
            "name": name,
            "chains": chains,
            "output": output
        }
    ]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # antibody & antigen structures prediction & sequence design
    predictor = TCRpMHCPredictor(
        ppi_path=esm_ppi_650m_tcr(),
        trunk_path=tfold_tcr_pmhc_trunk()
    )
    predictor.to(device)

    if torch.cuda.is_bf16_supported():
        predictor.to(torch.bfloat16)

    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"], chunk_size=256)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # test_fasta()
    # test_pmhc_example()
    # test_tcr_example()
    test_tcr_pmhc_example()
