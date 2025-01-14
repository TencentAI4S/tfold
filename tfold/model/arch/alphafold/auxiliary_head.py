# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/1/2 17:28
import torch.nn as nn

from .module.head import (DistogramHead, ExperimentallyResolvedHead, MaskedMSAHead, LDDTHead, TMScoreHead)


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()
        self.plddt = LDDTHead(config.lddt.c_in,
                              c_hidden=config.lddt.c_hidden,
                              num_bins=config.lddt.no_bins
                              )
        self.distogram = DistogramHead(config.distogram.c_z,
                                       num_bins=config.distogram.no_bins
                                       )
        self.masked_msa = MaskedMSAHead(
            config.masked_msa.c_m,
            num_classes=config.masked_msa.c_out
        )
        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config.experimentally_resolved
        )
        self.tm_enabled = config.tm.enabled
        if self.tm_enabled:
            self.tm = TMScoreHead(config.tm.c_z, config.tm.no_bins)

    def forward(self, outputs):
        aux_out = {}
        aux_out["lddt_logits"] = self.plddt(outputs["sm_single"])
        aux_out["distogram_logits"] = self.distogram(outputs["pair"])
        aux_out["experimentally_resolved_logits"] = self.experimentally_resolved(outputs["single"])
        aux_out["msa_logits"] = self.masked_msa(outputs["msa"])

        if self.tm_enabled:
            aux_out["tm_logits"] = self.tm(outputs["pair"])

        return aux_out
