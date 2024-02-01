# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import argparse
import os
from glob import glob

from tfold.protein.parser.a3m_parser import merge_a3ms, a3m_to_a2m
from tfold.protein.tool import ColabSearch


def generate_msa(fasta_file,
                 db1_path='uniref30_2103_db',
                 db2_path='pdb70',
                 db3_path='colabfold_envdb_202108_db',
                 output_dir=None,
                 num_workers=1,
                 mmseqs='mmseqs'):
    """Generate MSA using MMseqs2."""
    if output_dir is None:
        output_dir = os.path.dirname(fasta_file)

    colab_search = ColabSearch(output_dir,
                               db1_path=db1_path,
                               db2_path=db2_path,
                               db3_path=db3_path,
                               use_env=True,
                               use_templates=False,
                               num_threads=num_workers,
                               mmseqs=mmseqs)
    colab_search.run(fasta_file)
    # merge all a3m
    a3m_files = glob(f'{output_dir}/*.a3m')
    a3m_strs = []
    for a3m_file in a3m_files:
        a3m_str = open(a3m_file).read()
        a3m_strs.append(a3m_str)

    a3ms_str = merge_a3ms(a3m_strs)
    out_a3m_path = os.path.join(output_dir, 'target.a3m')
    with open(out_a3m_path, 'w') as f:
        f.write(a3ms_str)
    print('Merged A3M:', out_a3m_path)
    a2m_str = a3m_to_a2m(a3ms_str)
    out_a2m_path = os.path.join(output_dir, f'target.a2m')
    with open(out_a2m_path, 'w') as f:
        f.write(a2m_str)
    print('Merged A2M:', out_a2m_path)


def parse_args():
    database_dir = '/mnt/ai4x_ceph/fandiwu/buddy1/Datasets/colab_databases'
    db1 = f'{database_dir}/uniref30_2302_db'
    db2 = f'{database_dir}/pdb70'
    db3 = f'{database_dir}/colabfold_envdb_202108'
    parser = argparse.ArgumentParser(
        description='Generate MSA for structure predictions using MMseqs2'
    )
    parser.add_argument('--fasta_file', required=True, help='fasta file')
    parser.add_argument('--output_dir', default='target', help='output dir')
    parser.add_argument('--db1', default=db1, help='database 1 name')
    parser.add_argument('--db2', default=db2, help='database 2 name')
    parser.add_argument('--db3', default=db3, help='database 3 name')
    parser.add_argument('--mmseqs', default='/usr/bin/mmseqs', help='path of mmseqs binary')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    generate_msa(args.fasta_file,
                 args.db1,
                 args.db2,
                 args.db3,
                 output_dir=args.output_dir,
                 num_workers=args.num_workers,
                 mmseqs=args.mmseqs)
