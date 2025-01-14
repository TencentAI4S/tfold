![header](docs/tfold.png)

--------------------------------------------------------------------------------

English | [简体中文](./README-zh.md)

This package provides an implementation of the inference pipeline of tFold, including tFold-Ab, tFold-Ag and tFold-TCR.

![demo](docs/demo.png)

We also provide:

1. An pre-trained language named ESM-PPI, works to extract both the intra-chain and inter-chain
information of the protein complex to generate features for the down-streaming task.
2. The test set we construct in our paper.
3. A human germline antibody frameworks library to guide antibody generation using tFold-Ag.

Any publication that discloses findings arising from using this source code or the model parameters
should cite the tFold paper.

Please also refer to the Supplementary Information for a detailed description of the method.

If you have any questions, please contact the tFold team at fandiwu@tencent.com.

For business partnership opportunities, please contact leslielwang@tencent.com.

## Main models

|  **Shorthand**  |                     **Dataset**                  |                                                                           **Description**                                                                                 |
| :-------------: | :----------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     ESM-PPI     |          UniRef50, PDB, PPI, Antibody            | General-purpose protein language model, further pre-trained using ESM2 with 650M parameters. Can be used to predict multimer structure directly from individual sequences |
|     ESM-PPI-tcr |    UniRef50, PDB, PPI, Antibody, TCR, peptide    | General-purpose protein language model, further pre-trained using ESM2 with 650M parameters. Can be used to predict multimer structure directly from individual sequences |
|     tFold-Ab    |          SAbDab (before 31 December 2021)        |                                        SOTA antibody structure prediction model. MSA-free prediction with ESM-PPI                                                         |
|     tFold-Ag    |          SAbDab (before 31 December 2021)        |                            SOTA antibody-antigen complex structure prediction model. Can be used for virtual screening of binding antibodies and antibody design          |
|     tFold-TCR   |         STCRDab (before 31 December 2021)        |                                SOTA TCR-complex structure prediction model. MSA-free prediction with ESM-PPI. Can be used for TCR design                                  |

## OverView


### Main Results

#### Unbound Antibody Prediction (SAbDab-22H2-Ab)
|       **Model**       |  **RMSD-CDR-H3** | **DockQ**  |
| :------------------:  | :--------------: | :--------: |
|  AlphaFold-Multimer   |     3.09         |   0.769    |
|  IgFold               |     3.40         |   0.711    |
|  DeepAb               |     3.77         |   0.718    |
|  ImmuneBuilder        |     3.49         |   0.745    |
|  **tFold-Ab**         |     3.04         |   0.766    |

#### Unbound Nanobody Prediction (SAbDab-22H2-Nano)
|       **Model**       |  **RMSD-CDR-H3** |
| :------------------:  | :--------------: |
| AlphaFold             |     3.96         |
| IgFold                |     4.64         |
| ImmuneBuilder         |     3.79         |
| ESMFold               |     3.80         |
| OmegaFold             |     3.63         |
| **tFold-Ab**          |     3.57         |

#### Antibody-Antigen Complex Prediction (SAbDab-22H2-AbAg)
|       **Model**       |     **DockQ**    | **Success Rate**  |
| :------------------:  | :--------------: | :---------------: |
| AlphaFold-Multimer    |     0.162        |        18.7       |
| AlphaFold-3           |     0.265        |        33.3       |
| **tFold-Ag**          |     0.223        |        29.2       |

#### unbound TCR Prediction (STCRDab-22-TCR)
|       **Model**       |  **RMSD-CDR-A3** | **RMSD-CDR-B3** | **DockQ**  |
| :------------------:  | :--------------: | :-------------: | :--------: |
| AlphaFold-Multimer    |       1.89       |      1.62       |   0.785    |
| AlphaFold-3           |       1.80       |      1.50       |   0.769    |
| TCRModel2             |       1.77       |      1.52       |   0.795    |
| **tFold-TCR**         |       1.66       |      1.35       |   0.795    |

#### unbound pMHC Prediction (STCRDab-22-pMHC)
|       **Model**       | **DockQ**  |
| :------------------:  | :--------: |
| AlphaFold-Multimer    |  0.927     |
| AlphaFold-3           |  0.926     |
| **tFold-TCR**         |  0.908     |

#### TCR-pMHC Prediction (STCRDab-22-TCR_pMHC)
|       **Model**       | **DockQ**  |   **RMSD**   |  **Success Rate** |
| :------------------:  | :--------: | :----------: | :---------------: |
| AlphaFold-Multimer    |  0.490     |   3.601      |       83.3        |
| AlphaFold-3           |  0.496     |   3.094      |       72.2        |
| **tFold-TCR**         |  0.496     |   2.413      |       94.4        |

## Installation

###
1. Clone the package
```shell
git clone https://github.com/TencentAI4S/tfold.git
cd tfold
```

2. Prepare the environment

* Please follow the instructions in [INSTALL.md](./docs/INSTALL.md) to set up the environment

3. Download pre-trained weights under params directory (Optional)
    * [Tencent Weiyun Drive](https://share.weiyun.com/7kBaiMjY)
    * [Google Drive](https://drive.google.com/file/d/15xNTvccugK1gSPIjQqrTiN1fufVzdb0B/view?usp=sharing)
    * [Zenodo](https://zenodo.org/records/12602915)

**Note**:

If you download the weights in the folder `./checkpoints`, you can proceed directly with the following steps.

If you don't download the weights, the weights will be downloaded automatically when you run the code.
4. Download sequence databases for mas searching (only needed for tFold-Ag)
```shell
sh scripts/setup_database.sh
```
## Dataset
###

1. Test set we construct in our paper
  * [Tencent Weiyun Drive](https://share.weiyun.com/zycZDrfA)
  * [Google Drive](https://drive.google.com/file/d/1szSr5bjP3Y6XbhUpbfZEb9ZL9UMPXtvZ/view?usp=drive_link)

2. Human germline antibody frameworks library to guide antibody generation
  * [Tencent Weiyun Drive](https://share.weiyun.com/qgM7rhoM)
  * [Google Drive](https://drive.google.com/file/d/1dzNFUIk3Mt_1yPckvVaGll8IrxLJ82py/view?usp=sharing)


Our repository supports two methods of use, direct use or pip installation.
## Quick Start

You can use a fasta file (--fasta) or a json file (--json) as input.


### tFold-Ab
#### Example 1: predicting the structure of a antibody & nanobody using tFold-Ab
```
# antibody
python projects/tfold_ab/predict.py --fasta examples/fasta.files/7ox3_A_B.fasta --output examples/predictions/7ox3_A_B.pdb

# nanobody
python projects/tfold_ab/predict.py --fasta examples/fasta.files/7ocj_B.fasta --output examples/predictions/7ocj_B.pdb
```
### tFold-Ag

#### Example 1: predicting the structure of a antibody-antigen complex & nanobody-antigen complex with pre-computed MSA
```
# antibody-antigen complex
python projects/tfold_ag/predict.py --fasta examples/fasta.files/8df5_A_B_R.fasta --msa examples/msa.files/8df5_R.a3m --output examples/predictions/8df5_A_B_R.pdb

# nanobody-antigen complex
python projects/tfold_ag/predict.py --fasta examples/fasta.files/7sai_C_NA_A.fasta --msa examples/msa.files/7sai_A.a3m --output examples/predictions/7sai_C_NA_A.pdb

```

#### Example 2: Generate MSA for structure predictions using MMseqs2
```
python projects/tfold_ag/gen_msa.py --fasta_file=examples/fasta.files/PD-1.fasta --output_dir=examples/PD-1
```

#### Example 3: predicting the structure of a antibody-antigen complex & nanobody-antigen complex with inter-chain features
```
# generate inter-chain feature (ppi)
python projects/tfold_ag/gen_icf_feat.py --pid_fpath=examples/fasta.files/8df5_A_B_R.fasta --fas_dpath=examples/fasta.files/ --pdb_dpath=examples/pdb.files.native/ --icf_dpath=examples/icf.files.ppi --icf_type=ppi

# antibody-antigen complex prediction with inter-chain feature
python projects/tfold_ag/predict.py --fasta examples/fasta.files/8df5_A_B_R.fasta --msa examples/msa.files/8df5_R.a3m --icf examples/icf.files.ppi/8df5_A_B_R.pt --output examples/predictions/8df5_A_B_R.pdb --model_version ppi
```

#### Example 4: CDRs loop deisgn with tFold-Ag with pre-computed MSA
```
python projects/tfold_ag/predict.py --fasta examples/fasta.files/7urf_O_P_A.cdrh3.fasta --msa examples/msa.files/7urf_A.a3m --output examples/predictions/7urf_O_P_A.pdb
```

### tFold-TCR

#### Example 1: predicting the structure of a TCR complex
```
# TCR
python projects/tfold_tcr/predict.py --json examples/tcr_example.json --output examples/predictions/ --model_version TCR

# pMHC complex
python projects/tfold_tcr/predict.py --json examples/pmhc_example.json --output examples/predictions/ --model_version pMHC

# Complex
python projects/tfold_tcr/predict.py --json examples/tcr_pmhc_example.json --output examples/predictions/ --model_version Complex
```

## Quick Start with Pip Installation
```shell
  cd tfold
  pip install .
  ```
After pip install, you can load and use a pretrained model as follows:

### Extract cross-chain information using ESM-PPI
```shell
import torch
import tfold

# Download the pre-trained model
model_path = tfold.model.esm_ppi_650m_ab()

# Load the model
model = tfold.model.PPIModel.restore(model_path)

# Prepare antibody sequences (can be single or multiple sequences)
data = [
        'QVQLVQSGAEVKKPGASVKVSCKASGYPFTSYGISWVRQAPGQGLEWMGWISTYNGNTNYAQKFQGRVTMTTDTSTTTGYMELRRLRSDDTAVYYCARDYTRGAWFGESLIGGFDNWGQGTLVTVSS', # Heavy chain
        'EIVLTQSPGTLSLSPGERATLSCRASQTVSSTSLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQHDTSLTFGGGTKVEIK' # Light chain
]
ppi_output = model(data)

```
### Predict antibody structures with tFold-Ab
```shell
import torch
import tfold

# Download the pre-trained model
ppi_model_path = tfold.model.esm_ppi_650m_ab()
tfold_model_path = tfold.model.tfold_ab_trunk()

# Load the model
model = tfold.deploy.PLMComplexPredictor.restore_from_module(ppi_model_path, tfold_model_path)

# Prepare antibody sequences (can be single or multiple sequences)
data =[
        {
          "sequence": 'QVQLVQSGAEVKKPGASVKVSCKASGYPFTSYGISWVRQAPGQGLEWMGWISTYNGNTNYAQKFQGRVTMTTDTSTTTGYMELRRLRSDDTAVYYCARDYTRGAWFGESLIGGFDNWGQGTLVTVSS', # Heavy chain
          "id": 'H'
          },
        {
          "sequence": 'EIVLTQSPGTLSLSPGERATLSCRASQTVSSTSLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQHDTSLTFGGGTKVEIK', # Light chain
          "id": 'L'
          }]
output_path = '8df5_A_B_R.pdb'

model.infer_pdb(data, output_path)

```

### Predict the structure of a antibody-antigen complex with tFold-Ag
```shell
import torch
import tfold

# Download the pre-trained model of ESM-PPI
ppi_model_path = tfold.model.esm_ppi_650m_ab()
# Download the pre-trained model of alphaFold
alphafold_path  = tfold.model.alpha_fold_4_ptm()
# Download base model for tFold-Ag
tfold_model_path = tfold.model.tfold_ag_base()

# Download the ppi model for tFold-Ag
# tfold_model_path = tfold.model.tfold_ag_ppi()

# Load the model
model = tfold.deploy.AgPredictor(ppi_model_path, alphafold_path, tfold_model_path)

# Prepare antibody-antigen sequences
msa_path = 'examples/msa.files/8df5_R.a3m'
with open(msa_path) as f:
   msa, deletion_matrix = tfold.protein.parser.parse_a3m(f.read())

# if you don't have msa, you can use the following code to generate msa
#from projects.tfold_ag.gen_msa import generate_msa
#with open('8df5_R.fasta', 'w') as f:
#    f.write('>8df5_R\nMGILPSPGMPALLSLVSLLSVLLMGCVAETGTRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTHHHHHHHHGGSSGLNDIFEAQKIEWHE')
#generate_msa('8df5_R.fasta', output_dir='examples/msa.files/')
#with open('examples/msa.files/8df5_R.a3m') as f:
#   msa, deletion_matrix = tfold.protein.parser.parse_a3m(f.read())


data = [
         {
             "id": "H",
             "sequence": "QVQLVQSGAEVKKPGASVKVSCKASGYPFTSYGISWVRQAPGQGLEWMGWISTYNGNTNYAQKFQGRVTMTTDTSTTTGYMELRRLRSDDTAVYYCARDYTRGAWFGESLIGGFDNWGQGTLVTVSS"
         },
         {
             "id": "L",
             "sequence": "EIVLTQSPGTLSLSPGERATLSCRASQTVSSTSLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQHDTSLTFGGGTKVEIK"
         },
         {
             "id": "A",
             "sequence": "MGILPSPGMPALLSLVSLLSVLLMGCVAETGTRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVKGFNCYFPLQSYGFQPTYGVGYQPYRVVVLSFELLHAPATVCGPKKSTHHHHHHHHGGSSGLNDIFEAQKIEWHE",
             "msa": msa,
             "deletion_matrix": deletion_matrix
         }
        ]

output_path = '8df5_A_B_R.pdb'

model.infer_pdb(data, output_path)

```

### Predict TCR structures using tFold-TCR
```shell
import torch
import tfold

# Download the pre-trained model
ppi_model_path = tfold.model.esm_ppi_650m_tcr()
tfold_model_path = tfold.model.tfold_tcr_trunk()

# Load the model
model = tfold.deploy.TCRPredictor.restore_from_module(ppi_model_path, tfold_model_path)

# Prepare TCR sequences
data =[
            {
                "id": "B",
                "sequence": "NAGVTQTPKFQVLKTGQSMTLQCSQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSIRGSRGEQFFGPGTRLTVL"
            },
            {
                "id": "A",
                "sequence": "AQEVTQIPAALSVPEGENLVLNCSFTDSAIYNLQWFRQDPGKGLTSLLLIQSSQREQTSGRLNASLDKSSGRSTLYIAASQPGDSATYLCAVTNQAGTALIFGKGTTLSVSS"
            }
        ]

output_path = '6zkw_E_D_A_B_C.pdb'

model.infer_pdb(data, output_path)

```

### Predict TCR-pMHC structures using tFold-TCR
```shell
import torch
import tfold

# Download the pre-trained model
ppi_model_path = tfold.model.esm_ppi_650m_tcr()
tfold_model_path = tfold.model.tfold_tcr_pmhc_trunk()

# Load the model
model = tfold.deploy.TCRpMHCPredictor.restore_from_module(ppi_model_path, tfold_model_path)

# Prepare TCR-pMHC sequences
data =[
            {
                "id": "B",
                "sequence": "NAGVTQTPKFQVLKTGQSMTLQCSQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSIRGSRGEQFFGPGTRLTVL"
            },
            {
                "id": "A",
                "sequence": "AQEVTQIPAALSVPEGENLVLNCSFTDSAIYNLQWFRQDPGKGLTSLLLIQSSQREQTSGRLNASLDKSSGRSTLYIAASQPGDSATYLCAVTNQAGTALIFGKGTTLSVSS"
            },
            {
                "id": "M",
                "sequence": "GSHSLKYFHTSVSRPGRGEPRFISVGYVDDTQFVRFDNDAASPRMVPRAPWMEQEGSEYWDRETRSARDTAQIFRVNLRTLRGYYNQSEAGSHTLQWMHGCELGPDGRFLRGYEQFAYDGKDYLTLNEDLRSWTAVDTAAQISEQKSNDASEAEHQRAYLEDTCVEWLHKYLEKGKETLLHLEPPKTHVTHHPISDHEATLRCWALGFYPAEITLTWQQDGEGHTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPVTLRWKP"
            },
            {
                "id": "N",
                "sequence": "MIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
            },
            {
                "id": "P",
                "sequence": "RLPAKAPLL"
            }
        ]

output_path = '6zkw_E_D_A_B_C.pdb'

model.infer_pdb(data, output_path)

```


## Citing tFold

If you use tfold in your research, please cite our paper

```BibTeX
@article{wu2024fast,
  title={Fast and accurate modeling and design of antibody-antigen complex using tFold},
  author={Wu, Fandi and Zhao, Yu and Wu, Jiaxiang and Jiang, Biaobin and He, Bing and Huang, Longkai and Qin, Chenchen and Yang, Fan and Huang, Ningqiao and Xiao, Yang and others},
  journal={bioRxiv},
  pages={2024--02},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

and old version of tFold-Ab
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

Our new pre-print paper on tFold-TCR will be coming soon
