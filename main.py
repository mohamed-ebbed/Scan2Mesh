import torch
from data.shapenet import ShapeNet
from model.vertix_model import VertixModel
import tqdm
from util.visualization import visualize_pointcloud
from training import vertix_train


num_vertices=250

config = {
    'experiment_name': 'vertix_training',
    'device': 'cuda:0',  
    'is_overfit': False,
    'batch_size': 64,
    'resume_ckpt': None,
    'learning_rate': 0.001,
    'max_epochs': 100,
    'print_every_n': 10,
    'validate_every_n': 100,
    'sdf_path': 'data/shapenet_dim32_sdf',
    'meshes_path': 'data/shapenet_reduced',
    'class_mapping': 'data/shape_info.json',
    'split': 'train',
    'num_vertices': num_vertices
}
vertix_train.main(config)