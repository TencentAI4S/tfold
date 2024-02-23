![header](docs/tfold.png)

--------------------------------------------------------------------------------

[English](./README.md) | 简体中文

## 简介
本项目提供了 tFold-Ab 和 tFold-Ag 的推理流程的实现。

![demo](docs/demo.png)

我们还提供：

1. 名为ESM-PPI的预训练语言模型，用于提取蛋白质复合物的链内和跨链信息，为下游任务生成特征。
2. 论文中构建的测试集。
3. 人源化抗体框架模板库，用于指导使用tFold-Ag生成抗体。

任何公开使用此源代码或模型参数得出的发现的出版物都应引用tFold论文。

请参阅补充信息以获取方法的详细描述。

如果您有任何问题，请联系tFold团队，邮箱为ailab@tencent.com

## 主要模型
| 简称 | 数据集 | 描述 |
|------|--------|------|
| ESM-PPI | UniRef50, PDB, PPI, Antibody | 通用蛋白质语言模型，使用具有 650M 参数的 ESM2 进一步预训练。可直接从单个序列预测多聚体结构 |
| tFold-Ab | SAbDab (截止至2021年12月31日) | SOTA抗体结构预测模型。使用ESM-PPI 进行单序列复合物预测 |
| tFold-Ag | SAbDab (截止至2021年12月31日) | SOTA抗体抗原复合物结构预测模型。可用于虚拟筛选结合抗体和抗体设计 |

## 开始

###
1. Clone the package
```shell
git clone https://github.com/TencentAI4S/tFold.git
cd tFold
```

2. 安装环境
请参考[INSTALL.md](docs%2FINSTALL.md)

3. 下载模型
    * [Tencent Weiyun Drive](https://share.weiyun.com/EF0QKLj8)
    * [Google Drive](https://drive.google.com/file/d/1BRBsxSLaUAas8K0SMXiZdNaMARwNnRFN/view?pli=1)

4. 下载序列库用于构建MSA
```shell
sh scripts/setup_database.sh
```

## 数据集
###

1. 论文中构建的测试集
  * [Tencent Weiyun Drive](https://share.weiyun.com/e4byANXL)
  * [Google Drive](https://drive.google.com/file/d/1R-u3fkNxSOIG8bUmXbxTzPLSSj5iSWip/view?usp=sharing)

2. 人源化模板库，用于指导tFold-Ag生成抗体
  * [Tencent Weiyun Drive](https://share.weiyun.com/qgM7rhoM)
  * [Google Drive](https://drive.google.com/file/d/1dzNFUIk3Mt_1yPckvVaGll8IrxLJ82py/view?usp=sharing)


## 测试样例
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

If you use tFold in your research, please cite our paper

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
