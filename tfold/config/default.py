# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from ml_collections import FieldReference

from tfold.utils import Registry
from .config_node import CfgNode as CN

CONFIG_REGISTRY = Registry('config')

_C = CN()
_C.model = CN()
# default model
_C.model.arch = 'tfold_ag'


@CONFIG_REGISTRY.register('tfold_ab')
def complex_structure_model(_C):
    c_s = FieldReference(192)
    c_z = FieldReference(128)
    init_c_s = FieldReference(1280)
    init_c_z = FieldReference(660)
    _C.model.c_s = c_s
    _C.model.c_z = c_z
    _C.model.init_c_s = init_c_s
    _C.model.init_c_z = init_c_z
    _C.model.position_embedding_dim = 64
    _C.model.use_residue_embedding = False
    _C.model.tm_enabled = True
    _C.model.num_2d_layers = 16
    _C.model.num_3d_layers = 8
    _C.model.num_recycles = 1
    _C.model.num_2d_recycles = 1
    _C.model.num_3d_recycles = 8
    return _C


@CONFIG_REGISTRY.register()
def tfold_ag(_C):
    _C.model.plm = CN()
    c_s = FieldReference(192)
    c_z = FieldReference(128)
    init_c_s = FieldReference(1280)
    init_c_z = FieldReference(660)
    _C.model.plm.c_s = init_c_s
    _C.model.plm.c_z = init_c_z
    _C.model.ligand = CN()
    _C.model.ligand.c_s = c_s
    _C.model.ligand.c_z = c_z
    _C.model.ligand.init_c_s = init_c_s
    _C.model.ligand.init_c_z = init_c_z

    _C.model.ligand.position_embedding_dim = 64
    _C.model.ligand.use_residue_embedding = False
    _C.model.ligand.tm_enabled = True

    _C.model.ligand.num_2d_layers = 16
    _C.model.ligand.num_3d_layers = 8

    _C.model.ligand.num_recycles = 1
    _C.model.ligand.num_2d_recycles = 1
    _C.model.ligand.num_3d_recycles = 8

    _C.model.docking = CN()
    _C.model.docking.ligand_c_s = c_s
    _C.model.docking.ligand_c_z = c_z
    _C.model.docking.receptor_c_s = 256
    _C.model.docking.receptor_c_z = 128
    _C.model.docking.use_icf = False

    _C.model.docking.num_2d_layers = 32
    _C.model.docking.num_3d_layers = 8

    _C.model.docking.num_recycles = 1
    _C.model.docking.num_2d_recycles = 1
    _C.model.docking.num_3d_recycles = 8
    return _C
