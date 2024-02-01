# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import os
import sys
import argparse
import torch

sys.path.append('.')

from tfold import AgPredictor
from tfold.utils import setup


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody & antigen structures prediction and design w/ tFold-Ag')
    parser.add_argument('--pid_fpath', type=str, required=True, help='Path to the plain-text file of antibody IDs')
    parser.add_argument('--fas_dpath', type=str, required=True, help='Directory path to input antibody FASTA files')
    parser.add_argument(
        '--msa_fpath', type=str, required=True, help='Path to the pre-computed antigen MSA files (.a3m)'
    )
    parser.add_argument(
        '--pdb_dpath',
        type=str,
        default='pdb.files.tfold_ag',
        help='Directory path to output PDB files, default is "pdb.files.tfold_ag"',
    )
    parser.add_argument(
        '--icf_dpath',
        default=None,
        help='Directory path to inter-chain feature files (.pt)',
    )
    parser.add_argument(
        '--mdl_dpath', type=str, default='params', help='Path to the model directory, default is "params"'
    )
    parser.add_argument(
        '--model_ver',
        default='base',
        choices=['base', 'ppi', 'contact'],
        help='Specifies the tFold-Ag model version to use. '
             'Options are "base", "ppi", or "contact". The default is "base". '
             'Note: '
             'Use "base" for the default tFold-Ag model. '
             'Use "ppi" for prediction and design with epitope and paratope features using the tFold-Ag-ppi model. '
             'Use "contact" for prediction and design with contact features using the tFold-Ag-contact model.',
    )
    args = parser.parse_args()

    return args


def predict(pid_fpath, fas_dpath, msa_fpath, mdl_dpath, pdb_dpath='.', icf_dpath=None, model_ver='base'):
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""
    model_dir = mdl_dpath
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert os.path.exists(msa_fpath), (
        'MSA file is not existing, you can generate one using the gen_msa.py script '
        'found in the projects/tfold_ag directory.'
    )
    ag_fname = 'tfold_ag' if model_ver == 'base' else f'tfold_ag_{model_ver}'

    # antibody & antigen structures prediction & sequence design
    predictor = AgPredictor(
        ppi_path=f'{model_dir}/esm_ppi_650m.pth',
        ag_path=f'{model_dir}/{ag_fname}.pth',
        psp_path=f'{model_dir}/alphafold_4_ptm.pth',
    ).to(device)
    predictor.run_batch(pid_fpath, fas_dpath, msa_fpath, pdb_dpath, icf_dpath, chunk_size=None)


def main():
    args = parse_args()
    setup(True)
    predict(args.pid_fpath, args.fas_dpath,
            args.msa_fpath, args.mdl_dpath,
            pdb_dpath=args.pdb_dpath,
            icf_dpath=args.icf_dpath,
            model_ver=args.model_ver)


if __name__ == '__main__':
    main()
