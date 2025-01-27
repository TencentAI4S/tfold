# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import argparse
import os
import sys

import torch
import tqdm

sys.path.append('.')

from tfold.deploy import PLMComplexPredictor
from tfold.protein.parser import parse_fasta
from tfold.utils import setup, jload
from tfold.model.pretrain import tfold_ab_trunk, esm_ppi_650m_ab


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody structures prediction w/ tFold-Ab')
    parser.add_argument('--fasta', '-f',
                        type=str,
                        default=None,
                        help='path to input FASTA files for single inference')
    parser.add_argument('--json', '-j', type=str, default=None, help='json file for batch inference')
    parser.add_argument(
        '--output',
        type=str,
        default="examples/results/7ox3_A_B.pdb",
        help='directory of output pdb files when batching inference',
    )
    parser.add_argument(
        '--device', '-d', type=str, default=None, help='inference device'
    )
    parser.add_argument(
        '--chunk_size', '-cs',
        type=int,
        default=None,
        help='chunk size for long chain inference',
    )
    args = parser.parse_args()

    return args


def predict(args):
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    if args.fasta is not None:
        path = args.fasta
        sequences, ids, _ = parse_fasta(path)
        assert len(sequences) == 2 or 1, f"only support two chains in fasta file in antibody and one chain in nanobody"
        chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids)]
        save_dir, basename = os.path.split(path)
        name = basename.split(".")[0]
        output = args.output or f"{save_dir}/{name}.pdb"
        batches = [
            {
                "name": name,
                "chains": chains,
                "output": output
            }
        ]
    else:
        tasks = jload(args.json)
        batches = []
        for task in tasks:
            name = task["name"]
            task["output"] = f"{args.output}/{name}.pdb"
            batches.append(task)

    if args.device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)

    # antibody & antigen structures prediction
    print("> loading model...")
    predictor = PLMComplexPredictor.restore_from_module(ppi_path=esm_ppi_650m_ab(),
                                                        trunk_path=tfold_ab_trunk())
    predictor.to(device)

    chunk_size = args.chunk_size
    print(f"#inference samples: {len(batches)}")
    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"], chunk_size=chunk_size)


def main():
    args = parse_args()
    setup()
    predict(args)


if __name__ == '__main__':
    main()
