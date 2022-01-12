from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


from model.vertix_edge_model import VertixEdge
from data.shapenet import ShapeNet

from chamferdist import ChamferDistance

#from pytorch3d.loss import chamfer_distance

def train(model, train_dataloader, val_dataloader, device, config):



    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Here, we follow the original implementation to also use a learning rate scheduler -- it simply reduces the learning rate to half every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

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
    train_vertices_loss_running = 0.
    train_edges_loss_running = 0.


    chamfer_loss = ChamferDistance()

    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device, set optimizer gradients to zero, perform forward pass
            ShapeNet.move_batch_to_device(batch, device)
            optimizer.zero_grad()
            mask = batch["input_mask"]

            vertices , edges = model(batch['input_sdf'].float(), mask, x_indices, y_indices, batch['edges_adj'].float())


            # Mask out known regions -- only use loss on reconstructed, previously unknown regions
            ###
            
            target_vertices = batch['target_vertices']
            target_edges = batch['target_edges']

            target_vertices[mask != 1] = 0

            # Compute loss, Compute gradients, Update network parameters
            #########
            loss_vertices = chamfer_loss(vertices, target_vertices)

            loss_edges = cross_entropy(edges, target_edges.long())

            loss = loss_vertices + loss_edges

            loss.backward()
            
            optimizer.step()

            # Logging
            train_loss_running += loss.item()
            train_vertices_loss_running += loss_vertices.item()
            train_edges_loss_running += loss_edges.item()

            iteration = epoch * len(train_dataloader) + batch_idx

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}, train_vertices_loss: {train_vertices_loss_running / config["print_every_n"]:.6f}, train_edges_loss: {train_edges_loss_running / config["print_every_n"]:.6f}')
                train_loss_running = 0.
                train_vertices_loss_running = 0.
                train_edges_loss_running = 0.

            # Validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                # Set model to eval
                model.eval()

                # Evaluation on entire validation set
                loss_val = 0.
                loss_vertices_val = 0.
                loss_edges_val = 0.

                for batch_val in val_dataloader:
                    ShapeNet.move_batch_to_device(batch_val, device)

                    with torch.no_grad():

                        mask = batch_val["input_mask"]

                        
                        vertices , edges = model(batch_val['input_sdf'].float(), mask, x_indices, y_indices, batch_val['edges_adj'].float())

                        target_vertices = batch_val['target_vertices']
                        target_edges = batch_val['target_edges']


                        target_vertices[mask != 1] = 0

                    loss_vertices = chamfer_loss(vertices, target_vertices)
                    loss_edges  = cross_entropy(edges, target_edges.long())
                    loss_vertices_val += loss_vertices
                    loss_edges_val += loss_edges
                    loss_val += loss_vertices + loss_edges

                loss_val /= len(val_dataloader)
                loss_vertices_val /= len(val_dataloader)
                loss_edges_val /= len(val_dataloader)

                if loss_val < best_loss_val:
                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')
                    best_loss_val = loss_val

                print(f'[{epoch:03d}/{batch_idx:05d}] loss_vertices: {loss_vertices_val:.6f} | loss_edges: {loss_edges_val:.6f} | val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f}')

                # Set model back to train
                model.train()

        scheduler.step()


def main(config):

    torch.autograd.set_detect_anomaly(True)

    # Declare device

    device = config["device"]

    print("Device: {}".format(device))


    # Create Dataloaders
    train_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "train" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"],num_trajectories=config["num_trajectories"])

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

    val_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "val" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"],num_trajectories=config["num_trajectories"])

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
    model = VertixEdge(config["num_vertices"], config["feature_size"])

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)