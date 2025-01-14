# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/14 19:13
import os
import sys

import torch
import tqdm

sys.path.append('.')

from tfold import AgPredictor
from tfold.utils import jload, jdump
from tfold.protein.parser import parse_fasta
from tfold.protein.parser import parse_a3m
from tfold.model.pretrain import esm_ppi_650m_ab, tfold_ag_base, tfold_ag_ppi, alpha_fold_4_ptm


def write_demo_json():
    demo = [
        {
            "name": "8df5_A_B_R",
            "chains": [
                {
                    "id": "H",
                    "sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYPFTSYGISWVRQAPGQGLEWMGWISTYNGNTNYAQKFQGRVTMTTDTSTTTGYMELRRLRSDDTAVYYCARDYTRGAWFGESLIGGFDNWGQGTLVTVSS"
                },
                {
                    "id": "L",
                    "sequence": "EIVLTQSPGTLSLSPGERATLSCRASQTVSSTSLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQHDTSLTFGGGTKVEIK"
                },
                {
                    "id": "A",
                    "sequence": "MGILPSPGMPALLSLVSLLSVLLMGCVAETGTRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTHHHHHHHHGGSSGLNDIFEAQKIEWHE"
                }
            ]
        }
    ]
    msa_path = f"./examples/msa.files/8df5_R.a3m"
    with open(msa_path) as f:
        msa, deletion_matrix = parse_a3m(f.read())
    demo[0]["chains"][-1]["msa"] = msa
    demo[0]["chains"][-1]["deletion_matrix"] = deletion_matrix
    jdump(demo, "./examples/abag_example.json")


def test_json():
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    
    model_version = "base"
    output = "./examples/results"
    json_path = "./examples/abag_example.json"
    tasks = jload(json_path)
    batches = []
    for task in tasks:
        name = task["name"]
        task["output"] = f"{output}/{name}_test_json.pdb"
        batches.append(task)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # antibody & antigen structures prediction & sequence design
    predictor = AgPredictor(
        ppi_path=esm_ppi_650m_ab(),
        ag_path=tfold_ag_base() if model_version == "base" else tfold_ag_ppi(),
        psp_path=alpha_fold_4_ptm(),
    ).to(device)

    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"])


def test_fasta():
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    fasta_path = "./examples/fasta.files/8df5_A_B_R.fasta"
    msa_path = "./examples/msa.files/8df5_R.a3m"

    model_version = "base"
    output = "./examples/results/8df5_A_B_R_test_fasta.pdb"

    sequences, ids, _ = parse_fasta(fasta_path)
    assert len(sequences) in (2, 3), f"must be 2 chains in fasta file"
    chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids)]

    name = os.path.basename(fasta_path).split(".")[0]
    with open(msa_path) as f:
        msa, deletion_matrix = parse_a3m(f.read())

    if "A" in ids:
        idx = ids.index("A")
        assert chains[idx]["sequence"] == msa[0], f"A chain is not match msa"
        chains[idx]["msa"] = msa
        chains[idx]["deletion_matrix"] = deletion_matrix
    else:
        chains.append({"id": "A",
                       "sequence": msa[0],
                       "msa": msa,
                       "deletion_matrix": deletion_matrix})

    batches = [
        {
            "name": name,
            "chains": chains,
            "output": output
        }
    ]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # antibody & antigen structures prediction & sequence design
    predictor = AgPredictor(
        ppi_path=esm_ppi_650m_ab(),
        ag_path=tfold_ag_base() if model_version == "base" else tfold_ag_ppi(),
        psp_path=alpha_fold_4_ptm(),
    ).to(device)

    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"])


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # write_demo_json()
    test_json()
    # test_fasta()
