![header](docs/tfold.png)

--------------------------------------------------------------------------------

[English](./README.md) | 简体中文

## 简介
本项目提供了 tFold 系列工具的推理流程的实现，包括 tFold-Ab, tFold-Ag 和 tFold-TCR。

![demo](docs/demo.png)

我们还提供：

1. 名为ESM-PPI的预训练语言模型，用于提取蛋白质复合物的链内和跨链信息，为下游任务生成特征。
2. 论文中构建的测试集。
3. 人源化抗体框架模板库，用于指导使用tFold-Ag生成抗体。

任何公开使用此源代码或模型参数得出的发现的出版物都应引用tFold论文。

请参阅补充信息以获取方法的详细描述。

如果您有任何问题，请联系tFold团队，邮箱为 fandiwu@tencent.com。

商业合作，请联系商务团队，邮箱为 leslielwang@tencent.com。


## 主要模型
|   **简称**  |                     **数据集**             |                                     **描述**                                                      |
| :---------: | :----------------------------------------: | :---------------------------------------------------------------------------------------------:   |
| ESM-PPI     | UniRef50, PDB, PPI, Antibody               | 通用蛋白质语言模型，使用具有 650M 参数的 ESM2 进一步预训练。可直接从单个序列预测多聚体结构        |
| ESM-PPI-tcr | UniRef50, PDB, PPI, Antibody, TCR, peptide | 通用蛋白质语言模型，使用具有 650M 参数的 ESM2 进一步预训练。可直接从单个序列预测多聚体结构        |
| tFold-Ab    | SAbDab (截止至2021年12月31日)              | SOTA抗体结构预测模型。使用ESM-PPI 进行单序列复合物预测                                            |
| tFold-Ag    | SAbDab (截止至2021年12月31日)              | SOTA抗体抗原复合物结构预测模型。可用于虚拟筛选结合抗体和抗体设计                                  |
| tFold-TCR   | STCRDab (截止至2021年12月31日)             | SOTA TCR复合物结构预测模型。使用 ESM-PPI 进行单序列复合物预测，能够同时预测 TCR-pMHC 五条链的结构 |

### 主要结果

#### Unbound Antibody Prediction (SAbDab-22H2-Ab)
|       **Model**       |  **RMSD-CDR-H3** | **DockQ**  |
| :------------------:  | :--------------: | :--------: |
|  AlphaFold-Multimer   |     3.07         |   0.773    |
|  Chai-1               |     3.25         |   0.772    |
|  IgFold               |     3.37         |   0.715    |
|  DeepAb               |     3.73         |   0.721    |
|  ImmuneBuilder        |     3.46         |   0.749    |
|  **tFold-Ab**         |     3.01         |   0.770    |

#### Unbound Nanobody Prediction (SAbDab-22H2-Nano)
|       **Model**       |  **RMSD-CDR-H3** |
| :------------------:  | :--------------: |
| AlphaFold             |     3.96         |
| Chai-1                |     3.57         |
| IgFold                |     4.64         |
| ImmuneBuilder         |     3.79         |
| ESMFold               |     3.80         |
| OmegaFold             |     3.63         |
| **tFold-Ab**          |     3.57         |

#### Antibody-Antigen Complex Prediction (SAbDab-22H2-AbAg)
|       **Model**       |     **DockQ**    | **Success Rate**  |
| :------------------:  | :--------------: | :---------------: |
| AlphaFold-Multimer    |     0.158        |        18.2       |
| AlphaFold-3           |     0.257        |        32.3       |
| **tFold-Ag**          |     0.217        |        28.3       |

#### unliganded TCR Prediction (STCRDab-22-TCR)
|       **Model**       |  **RMSD-CDR-A3** | **RMSD-CDR-B3** | **DockQ**  |
| :------------------:  | :--------------: | :-------------: | :--------: |
| AlphFold-Multimer     |       1.89       |      1.62       |   0.785    |
| AlphaFold-3           |       1.80       |      1.50       |   0.769    |
| TCRModel2             |       1.77       |      1.52       |   0.795    |
| **tFold-TCR**         |       1.66       |      1.35       |   0.795    |

#### unbound pMHC Prediction (STCRDab-22-pMHC)
|       **Model**       | **DockQ**  |
| :------------------:  | :--------: |
| AlphFold-Multimer     |  0.927     |
| AlphaFold-3           |  0.926     |
| **tFold-TCR**         |  0.908     |

#### TCR-pMHC Complex Prediction (STCRDab-22-TCR_pMHC)
|       **Model**       | **DockQ**  |   **RMSD**   |  **Success Rate** |
| :------------------:  | :--------: | :----------: | :---------------: |
| AlphFold-Multimer     |  0.490     |   3.601      |       83.3        |
| AlphaFold-3           |  0.496     |   3.094      |       72.2        |
| **tFold-TCR**         |  0.496     |   2.413      |       94.4        |

## 开始

###
1. Clone the package
```shell
git clone https://github.com/TencentAI4S/tfold.git
cd tfold
```

2. 安装环境
请参考[INSTALL.md](docs%2FINSTALL.md)

3. 下载模型
    * [Tencent Weiyun Drive](https://share.weiyun.com/7kBaiMjY)
    * [Google Drive](https://drive.google.com/file/d/15xNTvccugK1gSPIjQqrTiN1fufVzdb0B/view?usp=sharing)
    * [Zenodo](https://zenodo.org/records/12602915)

**注意**：

如果您将权重下载到文件夹“./checkpoints”中，你可以直接运行后续的代码。

如果您不下载权重，则运行代码时将自动下载权重。

4. 下载序列库用于构建MSA
```shell
sh scripts/setup_database.sh
```

## 数据集
###

1. 论文中构建的测试集
  * [Tencent Weiyun Drive](https://share.weiyun.com/zycZDrfA)
  * [Google Drive](https://drive.google.com/file/d/1szSr5bjP3Y6XbhUpbfZEb9ZL9UMPXtvZ/view?usp=drive_link)

2. 人源化模板库，用于指导tFold-Ag生成抗体
  * [Tencent Weiyun Drive](https://share.weiyun.com/qgM7rhoM)
  * [Google Drive](https://drive.google.com/file/d/1dzNFUIk3Mt_1yPckvVaGll8IrxLJ82py/view?usp=sharing)


## 测试样例

你可以使用fasta文件或json文件作为输入，示例文件位于examples文件夹中。

### tFold-Ab
#### 示例一：使用tFold-Ab预测抗体结构和纳米体结构
```
# antibody
python projects/tfold_ab/predict.py --fasta examples/fasta.files/7ox3_A_B.fasta --output examples/predictions/7ox3_A_B.pdb

# nanobody
python projects/tfold_ab/predict.py --fasta examples/fasta.files/7ocj_B.fasta --output examples/predictions/7ocj_B.pdb
```
### tFold-Ag

#### 示例一：使用tFold-Ag预测抗体-抗原复合物和纳米体-抗原复合物
```
# antibody-antigen complex
python projects/tfold_ag/predict.py --fasta examples/fasta.files/8df5_A_B_R.fasta --msa examples/msa.files/8df5_R.a3m --output examples/predictions/8df5_A_B_R.pdb

# nanobody-antigen complex
python projects/tfold_ag/predict.py --fasta examples/fasta.files/7sai_C_NA_A.fasta --msa examples/msa.files/7sai_A.a3m --output examples/predictions/7sai_C_NA_A.pdb

```

#### 示例二：使用MMseqs2生成结构预测所需的MSA
```
python projects/tfold_ag/gen_msa.py --fasta_file=examples/fasta.files/PD-1.fasta --output_dir=examples/PD-1
```

#### 示例三：在提供inter-chain feature的情况下，预测抗体-抗原复合物以及纳米抗体-抗原复合物
```
# generate inter-chain feature (ppi)
python projects/tfold_ag/gen_icf_feat.py --pid_fpath=examples/fasta.files/8df5_A_B_R.fasta --fas_dpath=examples/fasta.files/ --pdb_dpath=examples/pdb.files.native/ --icf_dpath=examples/icf.files.ppi --icf_type=ppi

# antibody-antigen complex prediction with inter-chain feature
python projects/tfold_ag/predict.py --fasta examples/fasta.files/8df5_A_B_R.fasta --msa examples/msa.files/8df5_R.a3m --icf examples/icf.files.ppi/8df5_A_B_R.pt --output examples/predictions/8df5_A_B_R.pdb --model_version ppi
```

#### 示例四：使用预先计算的MSA通过tFold-Ag设计CDR区域
```
python projects/tfold_ag/predict.py --fasta examples/fasta.files/7urf_O_P_A.cdrh3.fasta --msa examples/msa.files/7urf_A.a3m --output examples/predictions/7urf_O_P_A.pdb
```

### tFold-TCR

#### 示例一：预测TCR复合物的结构
```
# TCR
python projects/tfold_tcr/predict.py --json examples/tcr_example.json --output examples/predictions/ --model_version TCR

# pMHC complex
python projects/tfold_tcr/predict.py --json examples/pmhc_example.json --output examples/predictions/ --model_version pMHC

# Complex
python projects/tfold_tcr/predict.py --json examples/tcr_pmhc_example.json --output examples/predictions/ --model_version Complex
```

## 使用Pip安装的使用案例

```shell
  cd tfold
  pip install .
  ```

### 使用ESM-PPI提取链间特征
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
### 使用tFold-Ab预测抗体结构
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

### 使用tFold-Ag预测抗体-抗原复合物结构
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

### 使用tFold-TCR预测TCR结构
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

### 使用tFold-TCR预测TCR-pMHC复合物结构
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

如果你在研究中使用了tFold, 请引用我们的工作

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

我们关于tFold-TCR的预印版文章也会很快上线
