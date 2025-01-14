# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/14 15:52
import os
import sys

import torch
import tqdm

sys.path.append('.')

from tfold.deploy import PLMComplexPredictor
from tfold.model.pretrain import tfold_ab
from tfold.protein.parser import parse_fasta
from tfold.utils import jload


def test_json():
    path = "../examples/ab_example.json"
    output = "../examples/results"
    tasks = jload(path)
    batches = []
    for task in tasks:
        name = task["name"]
        task["output"] = f"{output}/{name}_test_json.pdb"
        batches.append(task)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # antibody & antigen structures prediction
    print("> loading model...")
    predictor = PLMComplexPredictor.restore_from_hub("tfold_ab").to(device)
    if torch.cuda.is_bf16_supported():
        predictor.to(torch.bfloat16)

    print(f"#inference samples: {len(batches)}")
    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"], chunk_size=None)


def test_fasta():
    path = "../examples/fasta.files/7ox3_A_B.fasta"
    sequences, ids, _ = parse_fasta(path)
    assert len(sequences) == 2, f"only support two chains in fasta file"
    chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids)]
    save_dir, basename = os.path.split(path)
    name = basename.split(".")[0]
    output = "./examples/results/7ox3_A_B_test_fasta.pdb"
    batches = [
        {
            "name": name,
            "chains": chains,
            "output": output
        }
    ]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # antibody & antigen structures prediction
    print("> loading model...")
    predictor = PLMComplexPredictor.restore(tfold_ab()).to(device)
    if torch.cuda.is_bf16_supported():
        predictor.to(torch.bfloat16)

    print(f"#inference samples: {len(batches)}")
    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"])


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    test_fasta()
