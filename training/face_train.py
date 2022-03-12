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





import pickle

import gc

def train(model, train_dataloader, val_dataloader, device, config):



    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    x_indices = []
    y_indices = []

    for i in range(config["num_vertices"]):
        for j in range(config["num_vertices"]):
            x_indices.append(i)
            y_indices.append(j)

    # Set model to train
    model.train()

    best_loss_val = np.inf

    # Keep track of running average of train loss for printing
    train_loss_running = 0.


    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device, set optimizer gradients to zero, perform forward pass

            predicted_vertices, predicted_faces, edge_idx, hv, gt_faces = batch

            predicted_faces.float().requires_grad_(True)

            predicted_vertices.float().requires_grad_(True)


            edge_idx = edge_idx.to(config["device"]).squeeze().long().transpose(0,1)



            hv = hv.to(config["device"]).float()
            gt_faces = gt_faces.to(config["device"]).long()

            optimizer.zero_grad()

            mask = model(hv, edge_idx).argmax(1)


            output_faces = predicted_faces[mask == 1]

            gt_mesh = Meshes([predicted_vertices.squeeze()], [predicted_faces.squeeze()])
            output_mesh = Meshes([predicted_vertices.squeeze()], [output_faces.squeeze()])

            gt_points = sample_points_from_meshes(gt_mesh, num_samples=1000)
            output_points = sample_points_from_meshes(output_mesh, num_samples=1000)

            loss = chamfer_distance(output_points, gt_points)[0]

            loss.backward()
            
            optimizer.step()


            # Logging
            train_loss_running += loss.item()



            iteration = epoch * len(train_dataloader) + batch_idx


            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                #print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}, train_vertices_loss: {vertix_loss_running / config["print_every_n"]:.6f}, train_edges_loss: {edges_loss_running / config["print_every_n"]:.6f}')
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
               
                train_loss_running = 0.

                torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_last.ckpt')

            # Validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                # Set model to eval
                model.eval()

                # Evaluation on entire validation set
                loss_val = 0.
                loss_faces_val = 0.



                for batch_val in val_dataloader:
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

                    gt_points = sample_points_from_meshes(gt_mesh, num_samples=1000)
                    output_points = sample_points_from_meshes(output_mesh, num_samples=1000)

                    loss_faces = chamfer_distance(output_points, gt_points)[0]
                    

                    loss_faces_val += loss_faces
                    
                    loss_val += loss_faces


                loss_val /= len(val_dataloader)

                if loss_val < best_loss_val:
                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')
                    best_loss_val = loss_val
                torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_last.ckpt')


                #print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f} | loss_vertices: {loss_vertices_val:.6f} | loss_edges: {loss_edges_val:.6f}')
                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')

                model.train()


        scheduler.step()



def main(config):

    torch.autograd.set_detect_anomaly(True)

    # Declare device

    device = config["device"]

    print("Device: {}".format(device))


    # Create Dataloaders

    if not config.get('train_pickle'):
        train_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "train" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories=config["num_trajectories"], face_mode=True)
        train_dataset.filter_data()
        if config.get("overfit_single"):
            train_dataset.items = [train_dataset.items[1]]


    else:
        with open(config["train_pickle"], 'rb') as handle:
            train_dataset = pickle.load(handle)

    train_dataset.calculate_class_statistics()


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=0,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=train_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    if not config.get('val_pickle'):

        val_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "val" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories=config["num_trajectories"], face_mode=True)

        val_dataset.filter_data()

        if config.get("overfit_single"):

            val_dataset.items = [val_dataset.items[1]]


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

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not False:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)