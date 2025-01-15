# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.

import argparse
import os
import sys

sys.path.append('.')
import torch
import tqdm
from tfold.utils import setup, jload
from tfold.protein.parser import parse_fasta
from tfold.deploy import TCRPredictor, TCRpMHCPredictor, PeptideMHCPredictor
from tfold.model.pretrain import esm_ppi_650m_tcr, tfold_tcr_trunk, tfold_tcr_pmhc_trunk, tfold_pmhc_trunk


def parse_args():
    parser = argparse.ArgumentParser(description='Predict TCR-pMHC structures w/ tFold-TCR')
    parser.add_argument('--fasta', '-f', type=str, default=None,
                        help='Directory path to input FASTA files')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory path to output PDB files')
    parser.add_argument('--model_version', '-mv', type=str, default='TCR',
                        choices=['TCR', 'pMHC', 'Complex'],
                        help='tFold-TCR model version, TCR for TCR-only multimer prediction, '
                             'pMHC for pMHC multimer prediction, Complex for TCR-pMHC multimer prediction.'
                        )
    parser.add_argument('--json', '-j', type=str, default=None, help='json file for batch inference')
    parser.add_argument(
        '--chunk_size', '-cs',
        type=int,
        default=None,
        help='chunk size for long chain inference',
    )
    parser.add_argument(
        '--device', '-d', type=str, default=None, help='inference device'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='random seed for diversity sampling'
    )
    args = parser.parse_args()

    return args


def predict(args):
    """Predict antibody structures w/ pre-trained tFold-TCR models."""
    mv = args.model_version
    fasta_path = args.fasta
    if fasta_path is not None:
        sequences, ids, _ = parse_fasta(fasta_path)
        assert len(sequences) in (1, 2, 3, 4, 5), f"must be 1, 2 or 3 chains in fasta file"
        chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids)]
        save_dir, basename = os.path.split(fasta_path)
        name = basename.split(".")[0]
        output = args.output or f"{save_dir}/{name}_{mv}.pdb"
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
            task["output"] = f"{args.output}/{name}_{mv}.pdb"
            batches.append(task)

    if args.device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)

    if mv.startswith('TCR'):  # TCR multimer
        predictor = TCRPredictor.restore_from_module(ppi_path=esm_ppi_650m_tcr(),
                                                     trunk_path=tfold_tcr_trunk())
    elif mv.startswith('pMHC'):  # pMHC multimer
        predictor = PeptideMHCPredictor.restore_from_module(ppi_path=esm_ppi_650m_tcr(),
                                                            trunk_path=tfold_pmhc_trunk())
    else:  # TCR-pMHC multimer
        predictor = TCRpMHCPredictor(ppi_path=esm_ppi_650m_tcr(),
                                     trunk_path=tfold_tcr_pmhc_trunk())

    predictor.to(device)

    chunk_size = args.chunk_size
    print(f"#inference samples: {len(batches)}")
    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(task["chains"], filename=task["output"], chunk_size=chunk_size)


def main():
    args = parse_args()
    setup(True, seed=args.seed)
    predict(args)


if __name__ == '__main__':
    main()
