from pathlib import Path
import json

import numpy as np
import torch
import struct
import os
import trimesh
import tqdm


class ShapeNet(torch.utils.data.Dataset):

    num_classes = 8

    def __init__(self, sdf_path, meshes_path, class_mapping, split, threshold):

        super().__init__()
        self.sdf_path = sdf_path
        self.meshes_path = meshes_path
        #assert split in ['train', 'val', 'overfit']

        self.class_name_mapping = json.loads(Path(class_mapping).read_text())  # mapping for ShapeNet ids -> names
        self.classes = sorted(self.class_name_mapping.keys())

        self.truncation_distance = 3

        self.items = Path(f"data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

        self.threshold = threshold

    def __getitem__(self, index):

        sdf_id = self.items[index].split()[0]
        shape_id = sdf_id[:sdf_id.find("_")]

        input_sdf = self.get_shape_sdf(sdf_id)


        vertices, edges, faces, mask = self.get_shape_mesh(shape_id)

        

        input_sdf = np.minimum(np.abs(input_sdf),self.truncation_distance) * np.sign(input_sdf)

        steps=np.linspace(-1,1,32)
        grid = np.meshgrid(steps, steps, steps, indexing="ij")

        coordinates = np.zeros((32,32,32))
        
        xs = np.expand_dims(grid[0],0)
        ys = np.expand_dims(grid[1],0)
        zs = np.expand_dims(grid[2],0)

        input_sdf = np.expand_dims(input_sdf, 0)

        sign = np.sign(input_sdf)
    

        input_sdf = np.concatenate([input_sdf, sign, xs, ys, zs], axis=0)


        return {
            'name': f'{sdf_id}-{shape_id}',
            'input_sdf': input_sdf,
            'target_vertices': vertices,
            'input_mask': mask,
            #'target_faces': faces,
            #'target_edges': edges
        }

    def __len__(self):
        return len(self.items)


    @staticmethod
    def move_batch_to_device(batch, device):
        batch['input_sdf'] = batch['input_sdf'].to(device)
        batch['target_vertices'] = batch['target_vertices'].to(device)
        batch['input_mask'] = batch['input_mask'].to(device)
       # batch['target_faces'] = batch['target_faces'].to(device)
        #batch['target_edges'] = batch['target_edges'].to(device)

    def get_shape_sdf(self,shapenet_id):
        sdf = None
        file_name = shapenet_id + ".sdf"
        file_path = os.path.join(self.sdf_path, file_name)
        with open(file_path, "rb") as f:
            dims = np.fromfile(f, "uint64", count=3)
            sdf = np.fromfile(f, "float32")

            sdf = sdf.reshape(dims[0], dims[1], dims[2])

        return sdf

    def get_shape_mesh(self,shapenet_id):
        
        class_id = shapenet_id.split("/")[0]

        shape_id = shapenet_id.split("/")[1]


        file_name = f"{class_id}/{shape_id}/{shape_id}.obj"
        file_path = os.path.join(self.meshes_path, file_name)

        mesh = trimesh.load(file_path)
        vertices = mesh.vertices
        edges = mesh.edges
        faces = mesh.faces

        if vertices.shape[0] == self.threshold:
            mask = np.ones((1,self.threshold))
        else:
            to_add = self.threshold - vertices.shape[0]
            v=np.zeros((to_add,3))
            all_vertices=np.concatenate([vertices, v])
            mask=np.concatenate([np.ones(vertices.shape[0]), np.zeros(v.shape[0])],0)
            mask = mask.reshape((1,self.threshold))
            vertices = all_vertices

        mask = mask.squeeze()

        return np.array(vertices).astype(np.float32), np.array(edges), np.array(faces), mask


    def calculate_statistics(self, thresholds):

        print("Calculating statistics .. ")

        stats= [0]*len(thresholds)

        not_found = []


        for index in tqdm.trange(len(self.items)):
            sdf_id = self.items[index].split()[0]
            shapenet_id = sdf_id[:sdf_id.find("_")]

            class_id = shapenet_id.split("/")[0]

            shape_id = shapenet_id.split("/")[1]


            file_name = f"{class_id}/{shape_id}/{shape_id}.obj"
            file_path = os.path.join(self.meshes_path, file_name)

            if not os.path.exists(file_path):
                not_found.append(file_path)
                continue

            mesh = trimesh.load(file_path)
            vertices = np.array(mesh.vertices)

            num_vertices = vertices.shape[0]

            for i in range(len(thresholds)):

                threshold = thresholds[i]

                if num_vertices <= threshold:
                    stats[i] += 1

        print("Length of dataset: {}".format(self.__len__()))

        print("Data for each threshold:")


        for i in range(len(thresholds)):

            print("For threshold {} num of images {}".format(thresholds[i], stats[i]))

        print("{} images were not found".format(len(not_found)))


        return stats, not_found

    def filter_data(self):

        threshold = self.threshold

        filtered_items = []

        print("Length of dataset: {}".format(self.__len__()))


        print("Filtering data ..")


        for index in tqdm.trange(len(self.items)):

            sdf_id = self.items[index].split()[0]
            shapenet_id = sdf_id[:sdf_id.find("_")]

            class_id = shapenet_id.split("/")[0]

            shape_id = shapenet_id.split("/")[1]


            file_name = f"{class_id}/{shape_id}/{shape_id}.obj"
            file_path = os.path.join(self.meshes_path, file_name)

            if not os.path.exists(file_path):
                continue

            mesh = trimesh.load(file_path)
            vertices = np.array(mesh.vertices)

            num_vertices = vertices.shape[0]


            if num_vertices <= threshold:
                filtered_items.append(self.items[index])

        self.items = filtered_items

        print("Length of dataset: {}".format(self.__len__()))


    def calculate_class_statistics(self):

        classes_statistics = {}

        for index in tqdm.trange(len(self.items)):

            sdf_id = self.items[index].split()[0]
            shapenet_id = sdf_id[:sdf_id.find("_")]

            class_id = shapenet_id.split("/")[0]

            shape_id = shapenet_id.split("/")[1]

            if classes_statistics.get(class_id):
                classes_statistics[class_id] += 1

            else:
                classes_statistics[class_id] = 1



        for cls, cnt in classes_statistics.items():
            print("Class {} has {} shapes".format(cls, cnt))













