# -*- coding: utf-8 -*-

import pathlib
from skimage.io import imread

image_dir = pathlib.Path('/home/ishiyama/notebooks/keyakizaka/mobilenet/')

member_id_str = '001'
member_dir = image_dir / member_id_str
member_id = int(member_id_str)

