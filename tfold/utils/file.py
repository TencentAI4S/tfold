# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import io
import json
import os
import tempfile


def get_tmp_dpath():
    """Get the directory path to temporary files.

    Args: n/a

    Returns:
        tmp_dpath: directory path to temporary files
    """
    assert 'TMPDIR_PRFX' in os.environ, 'environmental variable <TMPDIR_PRFX> not defined'
    tmp_dpath_prfx = os.getenv('TMPDIR_PRFX')
    os.makedirs(os.path.dirname(tmp_dpath_prfx), exist_ok=True)
    tmp_dpath = tempfile.TemporaryDirectory(prefix=tmp_dpath_prfx).name  # pylint: disable=consider-using-with
    os.makedirs(tmp_dpath, exist_ok=True)

    return tmp_dpath


def jload(f, mode='r', object_pairs_hook=None):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f, object_pairs_hook=object_pairs_hook)
    f.close()
    return jdict
