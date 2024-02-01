"""Antibody & antigen structures prediction and design w/ pre-trained tFold-Ag models."""
import os
import argparse

import torch

from tfold.utils import setup_logger
from tfold.deploy import AgPredictor

# default directory path to pre-trained tFold-Ag models
MDL_DPATH = '/apdcephfs/share_1594716/fandiwu/Pre-trained.Models/tFold-Ag-models-release'


def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='Antibody & antigen structures prediction and design w/ tFold-Ag')
    parser.add_argument('--pid_fpath', type=str, required=True,
                        help='Path to the plain-text file of antibody IDs')
    parser.add_argument('--fas_dpath', type=str, required=True,
                        help='Directory path to input antibody FASTA files')
    parser.add_argument('--msa_dpath', type=str, required=True,
                        help='Directory path to the pre-computed antigen MSA files')
    parser.add_argument('--pdb_dpath', type=str, required=True,
                        help='Directory path to output PDB files (.pdb)')
    parser.add_argument('--icf_dpath', required=False, default=None,
                        help='Directory path to inter-chain feature files (.pt)')
    parser.add_argument('--mdl_dpath', type=str, default=MDL_DPATH, help='model directory path')
    parser.add_argument('--model_ver', type=str, default='base', choices=['base', 'ppi', 'contact'],
                        help='tFold-Ag model version (<base>, <ppi> OR <contact>), '
                             '<base> for base tFold-Ag model, <ppi> for prediction & design w/ ppi feature, '
                             '<contact> for prediction & design w/ contact feature.')
    args = parser.parse_args()

    return args


def predict(pid_fpath,
            fas_dpath,
            msa_dpath,
            icf_dpath,
            pdb_dpath,
            mdl_dpath,
            model_ver):
    """Predict antibody & antigen sequence and structures w/ pre-trained tFold-Ag models."""

    # configurations
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # get antibody IDs
    with open(pid_fpath, 'r', encoding='UTF-8') as i_file:
        prot_ids = [i_line.strip() for i_line in i_file]

    # antibody & antigen structures prediction & sequence design
    predictor = AgPredictor(mdl_dpath, model_ver, device)

    for prot_id in prot_ids:
        fas_fpath = os.path.join(fas_dpath, f'{prot_id}.fasta')
        antigen_name = '_'.join([prot_id.split('_')[0], prot_id.split('_')[3]])  # for SAbDab-22H2-Ab
        msa_fpath = os.path.join(msa_dpath, f'{antigen_name}.a3m')
        pdb_fpath = os.path.join(pdb_dpath, f'{prot_id}.pdb')
        icf_fpath = None if icf_dpath is None else os.path.join(icf_dpath, f'{prot_id}.pt')
        predictor(fas_fpath, msa_fpath, pdb_fpath, icf_fpath)


def main():
    setup_logger()
    args = parse_args()
    # predict antibody structures w/ pre-trained tFold-Ag models
    predict(args.pid_fpath, args.fas_dpath, args.msa_dpath, args.icf_dpath,
            args.pdb_dpath, args.mdl_dpath, args.model_ver)


if __name__ == '__main__':
    main()
