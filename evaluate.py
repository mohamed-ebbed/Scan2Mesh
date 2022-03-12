from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


from model.face_model import FaceModel

from data.shapenet import ShapeNet

from scipy.optimize import linear_sum_assignment

from scipy.spatial import distance_matrix

from torchgeometry.losses.focal import FocalLoss

from pytorch3d.structures import Meshes

from pytorch3d.loss import chamfer_distance

from pytorch3d.ops import sample_points_from_meshes

import tqdm





import pickle

import gc

def evaluate(model, val_dataloader, device, config):


    x_indices = []
    y_indices = []

    loss_faces_val = 0
    normal_sim_val = 0

    for i in range(config["num_vertices"]):
        for j in range(config["num_vertices"]):
            x_indices.append(i)
            y_indices.append(j)
  
    for batch_val in tqdm.tqdm(val_dataloader):
        predicted_vertices, predicted_faces, edge_idx, hv, gt_faces = batch_val

        predicted_faces.float().requires_grad_(True)

        predicted_vertices.float().requires_grad_(True)


        edge_idx = edge_idx.to(config["device"]).squeeze().long().transpose(0,1)



        hv = hv.to(config["device"]).float()
        gt_faces = gt_faces.to(config["device"]).long()

        with torch.no_grad():

            mask = model(hv, edge_idx).argmax(1)

        output_faces = predicted_faces[mask == 1]

        gt_mesh = Meshes([predicted_vertices.squeeze()], [predicted_faces.squeeze()])
        output_mesh = Meshes([predicted_vertices.squeeze()], [output_faces.squeeze()])

        gt_points, gt_normals = sample_points_from_meshes(gt_mesh, num_samples=1000,return_normals=True)
        output_points, output_normals = sample_points_from_meshes(output_mesh, num_samples=1000,return_normals=True)

        loss_faces, normal_sim = chamfer_distance(output_points, gt_points, x_normals=output_normals, y_normals=gt_normals)

        
        loss_faces_val += loss_faces
        
        normal_sim_val += normal_sim


    loss_faces_val /= len(val_dataloader)
    normal_sim_val /= len(val_dataloader)

    print(f'chamfer_loss: {loss_faces_val:.6f} | normal_sim: {normal_sim_val:.6f}')




def main(config):

    torch.autograd.set_detect_anomaly(True)

    # Declare device

    device = config["device"]

    print("Device: {}".format(device))

    if not config.get('val_pickle'):

        val_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "val" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories=config["num_trajectories"], face_mode=True)

        val_dataset.filter_data(classes=config["classes"])


    else:
        with open(config["val_pickle"], 'rb') as handle:
            val_dataset = pickle.load(handle)

    


    val_dataset.calculate_class_statistics()


    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config["batch_size"],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=0,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=val_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    # Instantiate model

    model = FaceModel(config["feature_size"])

    model.load_state_dict(torch.load(config['evaluate_checkpoint'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Start training
    evaluate(model, val_dataloader, device, config)