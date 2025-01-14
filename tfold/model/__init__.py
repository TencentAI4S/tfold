# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from .build import build_model
# auto register model here
from .arch import PPIModel, ComplexStructureModel, AgModel, BaseModel, ComplexLiteModel

from .pretrain import alpha_fold_4_ptm, esm_ppi_650m_ab, esm_ppi_650m_tcr, tfold_ab_trunk, tfold_ab, tfold_ag_base, tfold_ag_ppi, tfold_pmhc_trunk, tfold_tcr_pmhc_trunk, tfold_tcr_trunk
