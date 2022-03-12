from pathlib import Path

import numpy as np
import torch

from model.vertix_model import VertixModel
from data.shapenet import ShapeNet

import torch.nn as nn

from scipy.optimize import linear_sum_assignment

from scipy.spatial import distance_matrix


def train(model, train_dataloader, val_dataloader, device, config):

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Here, we follow the original implementation to also use a learning rate scheduler -- it simply reduces the learning rate to half every 20 epochs
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Set model to train
    model.train()

    best_loss_val = np.inf

    # Keep track of running average of train loss for printing
    train_loss_running = 0.

    vertix_loss = nn.L1Loss()


    for epoch in range(config['max_epochs']):
        train_loss_running = 0.

        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device, set optimizer gradients to zero, perform forward pass

            _, _, input_sdf, target_vertices, mask, target_edges, edges_adj = batch

            input_sdf = input_sdf.to(config["device"])
            target_vertices = target_vertices.to(config["device"])
            mask = mask.to(config["device"])
            target_edges = target_edges.to(config["device"])
            edges_adj = edges_adj.to(config["device"])
            optimizer.zero_grad()
            vertices , _ = model(input_sdf.float())

            # Mask out known regions -- only use loss on reconstructed, previously unknown regions
            ###
            

            # Compute loss, Compute gradients, Update network parameters
            #########

            batch_loss = 0

    
            for b in range(mask.shape[0]):
                target_size = int(mask[b].sum())
                
                cost = distance_matrix(vertices[b].clone().cpu().detach(), target_vertices[b].clone().cpu().detach())

                vertix_idx, target_idx = linear_sum_assignment(cost)

                loss = vertix_loss(vertices[b,vertix_idx], target_vertices[b, target_idx])

                batch_loss += loss

            batch_loss /= mask.shape[0]

            batch_loss.backward()
            
            optimizer.step()

            # Logging
            train_loss_running += batch_loss.item()
            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
                train_loss_running=0

            # Validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                # Set model to eval
                model.eval()

                # Evaluation on entire validation set
                loss_val = 0.
                num_shapes = 0

                for batch_val in val_dataloader:


                    _, _, input_sdf, target_vertices, mask, target_edges, edges_adj = batch_val

                    input_sdf = input_sdf.to(config["device"])
                    target_vertices = target_vertices.to(config["device"])
                    mask = mask.to(config["device"])
                    target_edges = target_edges.to(config["device"])
                    edges_adj = edges_adj.to(config["device"])

                    with torch.no_grad():
                        
                        vertices , _ = model(input_sdf.float())


                    batch_loss = 0

                    target_size = int(mask.sum())

                
                    for b in range(mask.shape[0]):
                        target_size = int(mask[b].sum())

                        cost = distance_matrix(vertices[b].clone().cpu().detach(), target_vertices[b].clone().cpu().detach())


                        vertix_idx, target_idx = linear_sum_assignment(cost)

                        loss = vertix_loss(vertices[b,vertix_idx], target_vertices[b, target_idx])

                        batch_loss += loss

                    batch_loss /= config["batch_size"]

                    loss_val += batch_loss

                loss_val /= len(val_dataloader)

                if loss_val < best_loss_val:
                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')
                    best_loss_val = loss_val

                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')

                # Set model back to train
                model.train()

        #scheduler.step()


def main(config):

    # Declare device

    device = config["device"]

    print("Device: {}".format(device))


    # Create Dataloaders
    train_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "train" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories=config["num_trajectories"])

    train_dataset.filter_data()

    train_dataset.calculate_class_statistics()


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=train_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    val_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "val" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories = config["num_trajectories"])

    val_dataset.filter_data()

    val_dataset.calculate_class_statistics()


    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=val_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    # Instantiate model
    model = VertixModel(config["num_vertices"])

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)