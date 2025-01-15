# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import argparse
import os
import sys

import torch
import tqdm

sys.path.append('.')

from tfold.deploy import AgPredictor
from tfold.utils import setup, jload
from tfold.protein.parser import parse_fasta
from tfold.protein.parser import parse_a3m
from tfold.model.pretrain import esm_ppi_650m_ab, tfold_ag_base, tfold_ag_ppi, alpha_fold_4_ptm


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody & antigen structures prediction and design w/ tFold-Ag')
    parser.add_argument('--fasta', '-f', type=str, default=None,
                        help='Directory path to input antibody FASTA files')
    parser.add_argument('--msa', '-m', type=str, default=None,
                        help='Path to the pre-computed antigen MSA files (.a3m)')
    parser.add_argument('--json', '-j', type=str, default=None, help='json file for batch inference')
    parser.add_argument(
        '--output',
        type=str,
        default='pdb.files.tfold_ag',
        help='Directory path to output PDB files, default is "pdb.files.tfold_ag"',
    )
    parser.add_argument(
        '--icf',
        default=None,
        help='Directory path to inter-chain feature files (.pt)',
    )
    parser.add_argument(
        '--model_version', '-mv',
        default='base',
        choices=['base', 'ppi', 'contact'],
        help='Specifies the tFold-Ag model version to use. '
             'Options are "base", "ppi", or "contact". The default is "base". '
             'Note: '
             'Use "base" for the default tFold-Ag model. '
             'Use "ppi" for prediction and design with epitope and paratope features using the tFold-Ag-ppi model. '
             'Use "contact" for prediction and design with contact features using the tFold-Ag-contact model.',
    )
    parser.add_argument(
        '--chunk_size', '-cs',
        type=int,
        default=256,
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
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    mv = args.model_version
    msa_path = args.msa
    fasta_path = args.fasta
    icf_path = args.icf

    if fasta_path is not None:
        sequences, ids, _ = parse_fasta(fasta_path)
        assert len(sequences) in (1, 2, 3), f"must be 1, 2 or 3 chains in fasta file"
        chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids)]
        save_dir, basename = os.path.split(fasta_path)
        name = basename.split(".")[0]
        output = args.output or f"{save_dir}/{name}_{mv}.pdb"
        assert os.path.exists(msa_path), (
            'MSA file is not existing, you can generate one using the gen_msa.py script '
            'found in the projects/tfold_ag directory.'
        )
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
                "output": output,
                "icf_path": icf_path,
            }
        ]
    else:
        tasks = jload(args.json)
        batches = []
        for task in tasks:
            name = task["name"]
            ids = [chain["id"] for chain in task["chains"]]
            assert "A" in ids, "antigen sequence should be used"
            idx = ids.index("A")
            if "msa_path" in task["chains"][idx]:
                with open(task["chains"][idx]["msa_path"]) as f:
                    msa, deletion_matrix = parse_a3m(f.read())
                task["chains"][idx]["msa"] = msa
                task["chains"][idx]["deletion_matrix"] = deletion_matrix

            task["output"] = f"{args.output}/{name}_{mv}.pdb"
            batches.append(task)

    if args.device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)

    # antibody & antigen structures prediction & sequence design
    predictor = AgPredictor(
        ppi_path=esm_ppi_650m_ab(),
        ag_path=tfold_ag_base() if mv == "base" else tfold_ag_ppi(),
        psp_path=alpha_fold_4_ptm(),
    )
    predictor.to(device)

    for task in tqdm.tqdm(batches):
        predictor.infer_pdb(
            task["chains"], filename=task["output"], icf_path=task["icf_path"], chunk_size=args.chunk_size)


def main():
    args = parse_args()
    setup(True, seed=args.seed)
    predict(args)


if __name__ == '__main__':
    main()
