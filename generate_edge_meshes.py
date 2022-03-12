import torch
import tqdm
import random
import numpy as np
import pickle

from training import edge_train

import argparse
import json

from data.shapenet import ShapeNet

from inference.inference_vertix_edge import InferenceHandlerVertixEdgeModel

import open3d


import os 

def export_mesh_to_obj(path, vertices, faces):
    to_write = ""
    faces += 1
    obj_file = open(path, "w+")

    for i in range(vertices.shape[0]):
        vx = vertices[i,:]
        l = f"v {vx[0]} {vx[1]} {vx[2]}\n"
        to_write += l
    for i in range(faces.shape[0]):
        face = faces[i,:]
        l = f"f {face[0]} {face[1]} {face[2]}\n"
        to_write += l

    obj_file.write(to_write)
    obj_file.close()



parser = argparse.ArgumentParser(description='Arguments for training scan2mesh')

parser.add_argument('--config', type=str, help='The experiment config', required=True)

args = parser.parse_args()


with open(args.config, 'r') as f:
  config = json.load(f)

inferer = InferenceHandlerVertixEdgeModel('runs/edge_model/model_best.ckpt', config["num_vertices"], config["feature_size"], config["device"])

train_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "train" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories=config["num_trajectories"])


train_dataset.filter_data()

val_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "val" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories=config["num_trajectories"])

val_dataset.filter_data()


x_indices = []
y_indices = []

graph = np.ones((1,config["num_vertices"], config["num_vertices"],1))

for i in range(config["num_vertices"]):
    for j in range(config["num_vertices"]):
        x_indices.append(i)
        y_indices.append(j)


for i in tqdm.trange(len(train_dataset)):
    shape_id, input_sdf, vertices, mask, target_edges, edges_adj = train_dataset[i]


      
    output_pointcloud, edges = inferer.infer_single(input_sdf, mask, x_indices,y_indices,edges_adj)
    
    faces = []
    

    num_vertices= config["num_vertices"]
    

    for i in range(num_vertices):
        for j in range(i+1,num_vertices):
            for k in range(j+1,num_vertices):
                if edges[i][j] and edges[j][k] and edges[k][i]:     
                    faces.append(np.array([i,j,k]).reshape(1,-1))
    faces = np.concatenate(faces,0)
            
    
    shape_id, instance_id = shape_id.split("/")

    if not os.path.exists("data/shapenet_edges/{}".format(shape_id)):
        os.mkdir("data/shapenet_edges/{}".format(shape_id))

    
    export_mesh_to_obj("data/shapenet_edges/{}/{}.obj".format(shape_id, instance_id),output_pointcloud.copy(),faces.copy())

for i in tqdm.trange(len(val_dataset)):
    shape_id, input_sdf, vertices, mask, target_edges, edges_adj = val_dataset[i]

    output_pointcloud, edges = inferer.infer_single(input_sdf, mask, x_indices,y_indices,edges_adj)
    
    faces = []    

    num_vertices= config["num_vertices"]

    for i in range(num_vertices):
        for j in range(i+1,num_vertices):
            for k in range(j+1,num_vertices):
                if edges[i][j] and edges[j][k] and edges[k][i]:     
                    faces.append(np.array([i,j,k]).reshape(1,-1))
    faces = np.concatenate(faces,0)
            
    

    shape_id, instance_id = shape_id.split("/")

    if not os.path.exists("data/shapenet_edges/{}".format(shape_id)):
        os.mkdir("data/shapenet_edges/{}".format(shape_id))

    
    export_mesh_to_obj("data/shapenet_edges/{}/{}.obj".format(shape_id, instance_id),output_pointcloud.copy(),faces.copy())
    

    