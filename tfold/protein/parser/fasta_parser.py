"""Protein-related utility functions."""
import gzip
import os
from collections import OrderedDict
from functools import partial


def parse_fasta(path):
    """Parse the FASTA file.

    Args:
        path: path to the FASTA file (could be GZIP-compressed)

    Returns:
        prot_id: protein ID (as in the commentary line)
        aa_seq: amino-acid sequence
    """
    open_fn = partial(gzip.open, mode='rt') if path.endswith('.gz') else partial(open, mode='r')
    with open_fn(path) as f:
        fasta_string = f.read()

    return parse_fasta_string(fasta_string)


def parse_fasta_string(fasta_string):
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif line.startswith('#'):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def parse_fas_file_mult(path, is_ordered=True):
    """Parse the FASTA file containing multiple chains.

    Args:
    * path: path to the FASTA file (can be GZIP-compressed)
    * is_ordered: (optional) whether the output dict is ordered as in the FASTA file

    Returns:
    * aa_seq_dict: (ordered) dict of (ID, sequence) pairs
    """

    # parse all the lines in the FASTA file
    assert os.path.exists(path), f'FASTA file does not exist: {path}'
    if not path.endswith('.gz'):
        with open(path, 'r', encoding='UTF-8') as i_file:
            i_lines = [i_line.strip() for i_line in i_file]
    else:
        with gzip.open(path, 'rt') as i_file:
            i_lines = [i_line.strip() for i_line in i_file]

    # build an ordered dict of (ID, sequence) pairs
    key_last = None
    aa_seq_dict = OrderedDict() if is_ordered else {}
    for i_line in i_lines:
        if i_line.startswith('>'):
            key_last = i_line[1:]
            aa_seq_dict[key_last] = ''
        else:
            assert key_last is not None, f'failed to get the protein ID in {path}'
            aa_seq_dict[key_last] += i_line

    return aa_seq_dict


def export_fas_file(prot_id, aa_seq, path):
    """Export the amino-acid sequence to a FASTA file.

    Args:
        prot_id: protein ID (as in the commentary line)
        aa_seq: amino-acid sequence
        path: path to the FASTA file
    """

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w', encoding='UTF-8') as o_file:
        o_file.write(f'>{prot_id}\n{aa_seq}\n')


def export_fas_file_mult(aa_seq_dict, path):
    """Export amino-acid sequences of multiple chains to a FASTA file.

    Args:
        aa_seq_dict: ordered dict of (ID, sequence) pairs
        path: path to the FASTA file

    Returns: n/a
    """
    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w', encoding='UTF-8') as o_file:
        for prot_id, aa_seq in aa_seq_dict.items():
            o_file.write(f'>{prot_id}\n{aa_seq}\n')
