![header](docs/tfold.png)

--------------------------------------------------------------------------------

English | [简体中文](./README-zh.md)

This package provides an implementation of the inference pipeline of tFold-Ab and tFold-Ag.

![demo](docs/demo.png)

We also provide:

1. An pre-trained language named ESM-PPI, works to extract both the intra-chain and inter-chain
information of the protein complex to generate features for the down-streaming task.
2. The test set we construct in our paper.
3. A human germline antibody frameworks library to guide antibody generation using tFold-Ag.

Any publication that discloses findings arising from using this source code or the model parameters
should cite the tFold paper.

Please also refer to the Supplementary Information for a detailed description of the method.

If you have any questions, please contact the tFold team at fandiwu@tencent.com

## Main models
| Shorthand | Dataset | Description  |
|---------------|---------|--------------|
| ESM-PPI   | UniRef50, PDB, PPI, Antibody | General-purpose protein language model, further pre-trained using ESM2 with 650M parameters. Can be used to predict multimer structure directly from individual sequences |
| tFold-Ab  | SAbDab (before 31 December 2021) | SOTA antibody structure prediction model. MSA-free prediction with ESM-PPI |
| tFold-Ag  | SAbDab (before 31 December 2021) | SOTA antibody-antigen complex structure prediction model. Can be used for virtual screening of binding antibodies and antibody design |

## Installation

###
1. Clone the package
```shell
git clone https://github.com/TencentAILabHealthcare/tFold.git
cd tFold
```

2. Prepare the environment

* Please follow the instructions in [INSTALL.md](./docs/INSTALL.md) to set up the environment

3. Download pre-trained weights under params directory
    * [Tencent Weiyun Drive](https://share.weiyun.com/EF0QKLj8)
    * [Google Drive](https://drive.google.com/file/d/1BRBsxSLaUAas8K0SMXiZdNaMARwNnRFN/view?pli=1)

4. Download sequence databases for mas searching (only needed for tFold-Ag)
```shell
sh scripts/setup_database.sh
```
## Dataset
###

1. Test set we construct in our paper
  * [Tencent Weiyun Drive](https://share.weiyun.com/e4byANXL)
  * [Google Drive](https://drive.google.com/file/d/1R-u3fkNxSOIG8bUmXbxTzPLSSj5iSWip/view?usp=sharing)

2. Human germline antibody frameworks library to guide antibody generation
  * [Tencent Weiyun Drive](https://share.weiyun.com/qgM7rhoM)
  * [Google Drive](https://drive.google.com/file/d/1dzNFUIk3Mt_1yPckvVaGll8IrxLJ82py/view?usp=sharing)


## Quick Start
### tFold-Ab
#### Example 1: predicting the structure of a antibody & nanobody using tFold-Ab
```
# antibody
python projects/tfold_ab/predict.py --pid_fpath=examples/prot_ids.ab.txt --fas_dpath=examples/fasta.files --pdb_dpath=examples/pdb.files.ab

# nanobody
python projects/tfold_ab/predict.py --pid_fpath=examples/prot_ids.nano.txt --fas_dpath=examples/fasta.files --pdb_dpath=examples/pdb.files.nano
```
### tFold-Ag

#### Example 1: predicting the structure of a antibody-antigen complex & nanobody-antigen complex with pre-computed MSA
```
# antibody-antigen complex
python projects/tfold_ag/predict.py --pid_fpath=examples/prot_ids.abag.txt --fas_dpath=examples/fasta.files --msa_fpath=examples/msa.files/8df5_R.a3m --pdb_dpath=examples/pdb.files.abag

# nanobody-antigen complex
python projects/tfold_ag/predict.py --pid_fpath=examples/prot_ids.nanoag.txt --fas_dpath=examples/fasta.files --msa_fpath=examples/msa.files/7sai_A.a3m --pdb_dpath=examples/pdb.files.nano
```
# antibody-antigen complex
python projects/tfold_ag/predict.py --pid_fpath=examples/prot_ids.abag.txt --fas_dpath=examples/fasta.files --msa_fpath=examples/msa.files/8df5_R.a3m --pdb_dpath=examples/pdb.files.abag

#### Example 2: Generate MSA for structure predictions using MMseqs2
```
python projects/tfold_ag/gen_msa.py --fasta_file=examples/fasta.files/PD-1.fasta --output_dir=examples/PD-1
```

#### Example 3: predicting the structure of a antibody-antigen complex & nanobody-antigen complex with inter-chain features
```
# generate inter-chain feature (ppi)
python projects/tfold_ag/gen_icf_feat.py --pid_fpath=examples/prot_ids.abag.txt --fas_dpath=examples/fasta.files --pdb_dpath=examples/pdb.files.native --icf_dpath=examples/icf.files.ppi --icf_type=ppi

# antibody-antigen complex prediction with inter-chain feature
python projects/tfold_ag/predict.py --pid_fpath=examples/prot_ids.abag.txt --fas_dpath=examples/fasta.files --msa_fpath=examples/msa.files/8df5_R.a3m --pdb_dpath=examples/pdb.files.abag --icf_dpath=examples/icf.files.ppi --model_ver=ppi
```
# generate inter-chain feature (ppi)
python projects/tfold_ag/gen_icf_feat.py --pid_fpath=examples/prot_ids.abag.txt --fas_dpath=examples/fasta.files --pdb_dpath=examples/pdb.files.native --icf_dpath=examples/icf.files.ppi --icf_type=ppi

#### Example 4: CDRs loop deisgn with tFold-Ag with pre-computed MSA
```
python projects/tfold_ag/predict.py --pid_fpath=examples/prot_ids.design.txt --fas_dpath=examples/fasta.files --msa_fpath=examples/msa.files/7urf_A.a3m --pdb_dpath=examples/pdb.files.design
```
## Citing tFold

If you use tfold in your research, please cite our paper
```BibTeX
@article{wu2022tfold,
  title={tFold-ab: fast and accurate antibody structure prediction without sequence homologs},
  author={Wu, Jiaxiang and Wu, Fandi and Jiang, Biaobin and Liu, Wei and Zhao, Peilin},
  journal={bioRxiv},
  pages={2022--11},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```
our new pre-print paper will coming soon
