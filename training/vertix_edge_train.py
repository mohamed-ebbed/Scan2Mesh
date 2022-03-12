from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


from model.vertix_edge_model import VertixEdge
from data.shapenet import ShapeNet

from scipy.optimize import linear_sum_assignment

from scipy.spatial import distance_matrix

from torchgeometry.losses.focal import FocalLoss



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

    train_loss_running = 0.
    train_vertices_loss_running = 0.
    train_edges_loss_running = 0.



    cross_entropy = nn.CrossEntropyLoss(ignore_index=-1,reduction='sum')


    vertix_loss = nn.L1Loss()


    for epoch in range(config['max_epochs']):
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device, set optimizer gradients to zero, perform forward pass

            input_sdf, target_vertices, mask, target_edges, edges_adj = batch

            input_sdf = input_sdf.to(config["device"])
            target_vertices = target_vertices.to(config["device"])
            mask = mask.to(config["device"])
            target_edges = target_edges.to(config["device"])
            edges_adj = edges_adj.to(config["device"])

            optimizer.zero_grad()

            vertices , edges = model(input_sdf.float(), mask, x_indices, y_indices, edges_adj.float())

            data_train = input_sdf

            target_vertices[mask != 1] = 0


            vertix_batch_loss = 0


            vertix_batch_loss = vertix_loss(vertices, target_vertices)

            loss_edges = cross_entropy(edges, target_edges.long())


            loss = vertix_batch_loss + loss_edges

            loss.backward()
            
            optimizer.step()


            # Logging
            train_loss_running += loss.item()
            train_edges_loss_running += loss_edges.item()

            iteration = epoch * len(train_dataloader) + batch_idx


            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{batch_idx:05d}] train_loss: {train_loss_running / config["print_every_n"]:.6f}')
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

                num_shapes = 0

                

                for batch_val in val_dataloader:

                    input_sdf, target_vertices, mask, target_edges, edges_adj = batch_val

                    input_sdf = input_sdf.to(config["device"])
                    target_vertices = target_vertices.to(config["device"])
                    mask = mask.to(config["device"])
                    target_edges = target_edges.to(config["device"])
                    edges_adj = edges_adj.to(config["device"])

                    data_val = input_sdf


                    with torch.no_grad():

                        
                        vertices , edges = model(input_sdf.float(), mask, x_indices, y_indices, edges_adj.float())


                    vertix_batch_loss = 0
                    edge_batch_loss = 0


                    edges_matched = torch.ones(mask.shape[0],config["num_vertices"], config["num_vertices"]).requires_grad_(False).to(config["device"]) * -1
  

                    vertix_batch_loss = vertix_loss(vertices, target_vertices)




                    loss_edges = cross_entropy(edges, target_edges.long())

                    
                    loss_edges_val += loss_edges

                    loss_val += loss_edges


                loss_val /= len(val_dataloader)
                loss_edges_val /= len(val_dataloader)

                if loss_val < best_loss_val:
                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')
                    best_loss_val = loss_val

                print(f'[{epoch:03d}/{batch_idx:05d}] val_loss: {loss_val:.6f} | best_loss_val: {best_loss_val:.6f} | loss_edges: {loss_edges_val:.6f}')

                # Set model back to train
                model.train()


        #scheduler.step()



def main(config):

    torch.autograd.set_detect_anomaly(True)

    # Declare device

    device = config["device"]

    print("Device: {}".format(device))


    # Create Dataloaders

    if not config.get('train_pickle'):
        train_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "train" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories=config["num_trajectories"])
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
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=train_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    if not config.get('val_pickle'):

        val_dataset = ShapeNet(sdf_path=config["sdf_path"],meshes_path=config["meshes_path"], class_mapping=config["class_mapping"], split = "val" if not config["is_overfit"] else "overfit", threshold=config["num_vertices"], num_trajectories=config["num_trajectories"])

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
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True,  # This is an implementation detail to speed up data uploading to the GPU
        # worker_init_fn=val_dataset.worker_init_fn  TODO: Uncomment this line if you are using shapenet_zip on Google Colab
    )

    # Instantiate model
    model = VertixEdge(config["num_vertices"], config["feature_size"])

    # Load model if resuming from checkpoint
    if config['resume_ckpt'] is not False:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # Move model to specified device
    model.to(device)

    # Create folder for saving checkpoints
    Path(f'runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(model, train_dataloader, val_dataloader, device, config)