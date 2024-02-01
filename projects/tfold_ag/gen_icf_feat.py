# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging
import os
import shutil
import argparse
from collections import OrderedDict

import torch

from tfold.protein import PdbParser
from tfold.protein.parser.fasta_parser import export_fas_file, parse_fas_file_mult
from tfold.protein.utils import calc_inter_contacts, calc_ppi_sites
from tfold.utils import get_tmp_dpath
from tfold.utils import setup


dist_thres = 10


def parse_args():
    parser = argparse.ArgumentParser(description='Generate inter-chain feature for tFold-Ag')
    parser.add_argument('--pid_fpath', type=str, required=True, help='Path to the plain-text file of antibody IDs')
    parser.add_argument('--fas_dpath', type=str, required=True, help='Directory path to input antibody FASTA files')
    parser.add_argument('--pdb_dpath', type=str, required=True, help='Directory path to PDB files')
    parser.add_argument(
        '--icf_dpath',
        type=str,
        default='icf.files',
        help='Directory path to output PDB files, default is "icf.files"',
    )
    parser.add_argument(
        '--icf_type',
        required=True,
        choices=['epitope', 'ppi', 'contact'],
        help='Inter-chain feature type for tFold-Ag. Options are "epitope", "ppi", or "contact".',
    )
    args = parser.parse_args()

    return args


def gen_icf_feat(pid_fpath, fas_dpath, pdb_dpath, icf_dpath, icf_type):
    """Generate inter-chain feature for tFold-Ag"""
    tmp_dpath = get_tmp_dpath()
    if not os.path.isdir(icf_dpath):
        os.mkdir(icf_dpath)
    with open(pid_fpath, 'r') as F:
        prot_ids = [item.strip() for item in F.readlines()]

    for prot_id in prot_ids:
        fas_fpath = os.path.join(fas_dpath, prot_id + '.fasta')
        pdb_fpath = os.path.join(pdb_dpath, prot_id + '.pdb')

        # inter-chain feature
        icf_fpath = os.path.join(icf_dpath, prot_id + '.pt')

        aa_seq_dict = parse_fas_file_mult(fas_fpath)
        aa_seq_dict = {k[-1]: v for k, v in aa_seq_dict.items()}

        # generate per-chain FASTA files
        for chain_id in ['H', 'L', 'A']:
            if chain_id not in aa_seq_dict:
                continue
            chain_id_ext = f'{prot_id}_{chain_id}'
            fas_fpath_chn = os.path.join(tmp_dpath, f'{chain_id_ext}.fasta')
            export_fas_file(chain_id_ext, aa_seq_dict[chain_id], fas_fpath_chn)

        # renumber the native PDB file
        prot_data = OrderedDict()
        for chain_id in ['H', 'L', 'A']:
            if chain_id not in aa_seq_dict:
                continue
            fas_fpath_chn = os.path.join(tmp_dpath, f'{prot_id}_{chain_id}.fasta')
            aa_seq, cord_tns, cmsk_mat, _, error_msg = PdbParser.load(
                pdb_fpath, fas_fpath=fas_fpath_chn, chain_id=chain_id
            )
            assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'
            prot_data[chain_id] = {'seq': aa_seq, 'cord': cord_tns, 'cmsk': cmsk_mat}

        if 'H' in prot_data and 'L' in prot_data:
            ligand_id = 'H-L'
            prot_data['H-L'] = {
                'seq': prot_data['H']['seq'] + prot_data['L']['seq'],
                'cord': torch.cat([prot_data['H']['cord'], prot_data['L']['cord']], dim=0),
                'cmsk': torch.cat([prot_data['H']['cmsk'], prot_data['L']['cmsk']], dim=0),
            }
        else:
            ligand_id = 'H'

        # calculate the PPI sites
        if icf_type == 'epitope' or icf_type == 'ppi':
            ppi_data = calc_ppi_sites(prot_data, [ligand_id, 'A'], dist_thres)
            torch.save(ppi_data, icf_fpath)
            if icf_type == 'epitope':  # set paratope to 0
                ppi_data[: len(prot_data[ligand_id]['seq'])] = 0
            torch.save(ppi_data, icf_fpath)
        else:
            # calculate the Contact
            contact_data = calc_inter_contacts(prot_data, [ligand_id, 'A'], dist_thres)
            torch.save(contact_data, icf_fpath)
        logging.info('Finish generate the inter-chain feature for %s', prot_id)

    shutil.rmtree(tmp_dpath)


if __name__ == '__main__':
    args = parse_args()
    setup(True)
    gen_icf_feat(args.pid_fpath, args.fas_dpath, args.pdb_dpath, args.icf_dpath, args.icf_type)
