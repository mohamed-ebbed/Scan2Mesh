from pathlib import Path
import json

import numpy as np
import torch
import struct
import os
import trimesh


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8

    def __init__(self, sdf_path, meshes_path, class_mapping, split):

        super().__init__()
        self.sdf_path = sdf_path
        self.meshes_path = meshes_path
        assert split in ['train', 'val', 'overfit']

        self.class_name_mapping = json.loads(Path(class_mapping).read_text())  # mapping for ShapeNet ids -> names
        self.classes = sorted(class_name_mapping.keys())

        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):

        sdf_id = self.items[index]
        shape_id = sdf_id[:sdf_id.find("_")]

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)


        vertices, edges, faces = ShapeNet.get_shape_mesh(shape_id)
        

        input_sdf = np.minimum(np.abs(input_sdf),self.truncation_distance) * np.sign(input_sdf)

        input_sdf = np.expand_dims(np.abs(input_sdf),0)

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_vertices': vertices,
            'target_faces': faces,
            'target_edges': edges
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['input_sdf'] = batch['input_sdf'].to(device)
        batch['target_mesh'] = batch['target_mesh'].to(device)

    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None
        file_name = shapenet_id + ".sdf"
        file_path = os.path.join(self.sdf_path, file_name)
        with open(file_path, "rb") as f:
            dims = np.fromfile(f, "uint64", count=3)
            sdf = np.fromfile(f, "float32")

            sdf = sdf.reshape(dims[0], dims[1], dims[2])

        return sdf

    @staticmethod
    def get_shape_mesh(shapenet_id):
        
        file_name = "{shapenet_id}/model_simplified.obj"
        file_path = os.path.join(self.meshes_path, file_name)
        mesh = trimesh.load(file_path)
        vertices = mesh.vertices
        edges = mesh.edges
        faces = mesh.faces
        
        
        return vertices, edges, faces

