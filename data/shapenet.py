from pathlib import Path
import json

import numpy as np
import torch
import struct
import os
import trimesh
import tqdm

from scipy.optimize import linear_sum_assignment

from scipy.spatial import distance_matrix


class ShapeNet(torch.utils.data.Dataset):

    num_classes = 8

    def __init__(self, sdf_path, meshes_path, class_mapping, split, threshold, num_trajectories, face_mode=False):

        super().__init__()
        self.sdf_path = sdf_path
        self.meshes_path = meshes_path
        #assert split in ['train', 'val', 'overfit']

        self.class_name_mapping = json.loads(Path(class_mapping).read_text())  # mapping for ShapeNet ids -> names
        self.classes = sorted(self.class_name_mapping.keys())

        self.truncation_distance = 3

        self.items = Path(f"data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

        self.threshold = threshold

        self.num_trajectories = num_trajectories

        self.face_mode = face_mode

        self.edges_path = "./data/shapenet_edges"


    def __getitem__(self, index):

        sdf_id = self.items[index].split()[0]
        shape_id = sdf_id[:sdf_id.find("_")]

        input_sdf = self.get_shape_sdf(sdf_id)

        original_sdf = input_sdf



        vertices, edges, faces, mask = self.get_shape_mesh(shape_id)

        if self.face_mode:
            predicted_vertices, predicted_edges, predicted_faces, mask = self.get_shape_mesh(shape_id, load_edges_mesh=self.face_mode)

        

        if not self.face_mode:

            input_sdf = np.minimum(np.abs(input_sdf),self.truncation_distance) * np.sign(input_sdf)

            steps=np.linspace(-1,1,32)

            grid = np.meshgrid(steps, steps, steps, indexing="ij")

            
            xs = np.expand_dims(grid[0],0)
            ys = np.expand_dims(grid[1],0)
            zs = np.expand_dims(grid[2],0)

            input_sdf = np.expand_dims(input_sdf, 0)

            sign = np.sign(input_sdf)

            input_sdf = np.abs(input_sdf)

            target_edges = np.zeros((self.threshold, self.threshold))
            edges_adj = np.ones((self.threshold, self.threshold,1))



            for face in faces:
                target_edges[face[0],face[1]] = 1
                target_edges[face[1],face[2]] = 1
                target_edges[face[2],face[0]] = 1

            target_size = int(sum(mask))

            for i in range(target_size,self.threshold):
                for j in range(target_size, self.threshold):
                    target_edges[i][j] = -1



            input_sdf = np.concatenate([input_sdf, sign, xs, ys, zs], axis=0)
            return shape_id, original_sdf, input_sdf, vertices, mask, target_edges, edges_adj

        else:


            edge_idx = trimesh.graph.face_adjacency(predicted_faces)

            hv = np.zeros((len(predicted_faces),7))

            for i in range(len(predicted_faces)):


                
                curr_face = predicted_faces[i]

                v1 = predicted_vertices[curr_face[0]]

                v2 = predicted_vertices[curr_face[1]]

                v3 = predicted_vertices[curr_face[2]]

                centroid = (v1+v2+v3) / 3

                radius = np.linalg.norm(v2 - v1) + np.linalg.norm(v3 - v2) + np.linalg.norm(v3 - v1)

                normal = np.cross(v2 - v1, v3-v2)


                hv[i] = np.array([centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], radius])

            


                    
            return predicted_vertices, predicted_faces, edge_idx, hv, faces


            

    def __len__(self):
        return len(self.items)


    def get_shape_sdf(self,shapenet_id):
        sdf = None
        file_name = shapenet_id + ".sdf"
        file_path = os.path.join(self.sdf_path, file_name)
        with open(file_path, "rb") as f:
            dims = np.fromfile(f, "uint64", count=3)
            sdf = np.fromfile(f, "float32")

            sdf = sdf.reshape(dims[0], dims[1], dims[2])

        return sdf

    def get_shape_mesh(self,shapenet_id,load_edges_mesh=False):
        
        class_id = shapenet_id.split("/")[0]

        shape_id = shapenet_id.split("/")[1]

        if not load_edges_mesh:
            file_name = f"{class_id}/{shape_id}/{shape_id}.obj"
            file_path = os.path.join(self.meshes_path, file_name)

        else:
            file_name = f"{class_id}/{shape_id}.obj"
            file_path = os.path.join(self.edges_path, file_name)

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

    def calculate_weights(self):
        num_zeros = 0
        num_ones = 0
        for i in tqdm.trange(self.__len__()):
            input_sdf, vertices, mask, target_edges, edges_adj = self.__getitem__(i)
            num_zeros += (target_edges == 0).sum()
            num_ones += (target_edges == 1).sum()

        total = num_zeros + num_ones

        return total / num_zeros, total / num_ones


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

            print("For threshold {} num of shapes {}".format(thresholds[i], stats[i]))

        print("{} shapes were not found".format(len(not_found)))


        return stats, not_found

    def filter_data(self):

        threshold = self.threshold

        filtered_items = []

        print("Length of dataset: {}".format(self.__len__()))


        print("Filtering data ..")


        for index in tqdm.trange(len(self.items)):


            sdf_id = self.items[index].split()[0]
            trajectory = int(sdf_id.split("__")[1])
            if trajectory >= self.num_trajectories:
                continue

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














