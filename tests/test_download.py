# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 15:25
import os.path

from tfold.utils import download_file

stero_chemical_url = "https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt"

cache_dir = os.path.expanduser("~/.tfold")
os.makedirs(cache_dir, exist_ok=True)
filename = f"{cache_dir}/stereo_chemical_props.txt"
download_file(stero_chemical_url, filename, overwrite=True)
