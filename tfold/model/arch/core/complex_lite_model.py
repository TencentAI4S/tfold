# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/14 11:10
from tfold.config import configurable
from .complex_structure_model import ComplexStructureModel
from .ppi_model import PPIModel
from ..base_model import BaseModel
from ...build import MODEL_REGISTRY


@MODEL_REGISTRY.register("ab_model")
@MODEL_REGISTRY.register()
class ComplexLiteModel(BaseModel):
    """plm based complex structure"""

    @classmethod
    def from_config(cls, cfg):
        return {
            "ppi_cfg": cfg.ppi,
            "trunk_cfg": cfg.trunk
        }

    @configurable
    def __init__(self, ppi_cfg, trunk_cfg):
        super().__init__()
        self.ppi = PPIModel(ppi_cfg)
        self.trunk = ComplexStructureModel(trunk_cfg)

    def forward(self,
                sequences,
                asym_id=None,
                num_recycles=None,
                num_3d_recycles=None,
                chunk_size=None):
        """Run the antibody structure predictor.
        """
        plm_outputs = self.ppi(sequences)
        s = plm_outputs['sfea']
        z = plm_outputs['pfea']
        outputs = self.trunk(sequences, s, z,
                             asym_id=asym_id,
                             num_recycles=num_recycles,
                             num_3d_recycles=num_3d_recycles,
                             chunk_size=chunk_size)
        return outputs
