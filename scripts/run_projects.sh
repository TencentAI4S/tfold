#!/bin/bash

# antibody
python projects/tfold_ab/predict.py --fasta examples/fasta.files/7ox3_A_B.fasta --output examples/predictions/7ox3_A_B.pdb

# nanobody
python projects/tfold_ab/predict.py --fasta examples/fasta.files/7ocj_B.fasta --output examples/predictions/7ocj_B.pdb

# antibody-antigen complex
python projects/tfold_ag/predict.py --fasta examples/fasta.files/8df5_A_B_R.fasta --msa examples/msa.files/8df5_R.a3m --output examples/predictions/8df5_A_B_R.pdb

# nanobody-antigen complex
python projects/tfold_ag/predict.py --fasta examples/fasta.files/7sai_C_NA_A.fasta --msa examples/msa.files/7sai_A.a3m --output examples/predictions/7sai_C_NA_A.pdb

# generate inter-chain feature (ppi)
python projects/tfold_ag/gen_icf_feat.py --pid_fpath=examples/fasta.files/8df5_A_B_R.fasta --fas_dpath=examples/fasta.files/ --pdb_dpath=examples/pdb.files.native/ --icf_dpath=examples/icf.files.ppi --icf_type=ppi

# antibody-antigen complex prediction with inter-chain feature
python projects/tfold_ag/predict.py --fasta examples/fasta.files/8df5_A_B_R.fasta --msa examples/msa.files/8df5_R.a3m --icf examples/icf.files.ppi/8df5_A_B_R.pt --output examples/predictions/8df5_A_B_R.pdb --model_version ppi

# design CDR H3
python projects/tfold_ag/predict.py --fasta examples/fasta.files/7urf_O_P_A.cdrh3.fasta --msa examples/msa.files/7urf_A.a3m --output examples/predictions/7urf_O_P_A.pdb

# TCR
python projects/tfold_tcr/predict.py --json examples/tcr_example.json --output examples/predictions/ --model_version TCR

# pMHC complex
python projects/tfold_tcr/predict.py --json examples/pmhc_example.json --output examples/predictions/ --model_version pMHC

# Complex
python projects/tfold_tcr/predict.py --json examples/tcr_pmhc_example.json --output examples/predictions/ --model_version Complex
