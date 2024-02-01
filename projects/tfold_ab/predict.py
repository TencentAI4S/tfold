# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import os
import sys
import argparse
import torch

sys.path.append('.')

from tfold import AbPredictor
from tfold.utils import setup


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody structures prediction w/ tFold-Ab')
    parser.add_argument('--pid_fpath', type=str, required=True, help='Path to the plain-text file of protein IDs')
    parser.add_argument('--fas_dpath', type=str, required=True, help='Directory path to input FASTA files')
    parser.add_argument(
        '--pdb_dpath',
        type=str,
        default=f'pdb.files.tfold_ab',
        help='Directory path to output PDB files, default is "pdb.files.tfold_ab"',
    )
    parser.add_argument(
        '--mdl_dpath', type=str, default='params', help='Path to the model directory, default is "params"'
    )
    args = parser.parse_args()

    return args


def predict(pid_fpath, fas_dpath, mdl_dpath, output=None):
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # antibody & antigen structures prediction
    predictor = AbPredictor(ppi_path=f'{mdl_dpath}/esm_ppi_650m.pth', ab_path=f'{mdl_dpath}/tfold_ab.pth').to(device)

    with open(pid_fpath, 'r', encoding='utf-8') as i_file:
        prot_ids = [i_line.strip() for i_line in i_file]

    for prot_id in prot_ids:
        fas_fpath = os.path.join(fas_dpath, f'{prot_id}.fasta')
        assert os.path.exists(fas_fpath), fas_fpath
        pdb_fpath = os.path.join(output, f'{prot_id}.pdb')
        predictor(fas_fpath, pdb_fpath)


def main():
    args = parse_args()
    setup()
    predict(args.pid_fpath, args.fas_dpath, args.mdl_dpath, output=args.pdb_dpath)


if __name__ == '__main__':
    main()
