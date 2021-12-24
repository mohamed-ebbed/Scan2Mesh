import pybullet as p
import pybullet_data as pd
import os
import tqdm
import json
from pathlib import Path


class_name_mapping = json.loads(Path("data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
classes = class_name_mapping.keys()
dataset_base = "data/ShapeNetCore.v2"

p.connect(p.DIRECT)

for cls in tqdm.tqdm(classes):
    curr_dir = os.path.join(dataset_base,f"{cls}")
    for shape in os.listdir(curr_dir):
        name_in = os.path.join(curr_dir,f"{shape}/models/model_normalized.obj")
        name_out = os.path.join(curr_dir,f"{shape}/models/model_simplified.obj")
        name_log = os.path.join(curr_dir,f"{shape}/models/log.txt")
        p.vhacd(name_in, name_out, name_log ,maxNumVerticesPerCH= 10,resolution=50000)