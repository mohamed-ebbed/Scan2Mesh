# Scan2Mesh

A non-official PyTorch implementation for **Scan2Mesh: From Unstructured Range Scans to 3D Meshes** paper on ShapeNet dataset

![Screenshot from 2022-01-09 16-58-47](https://user-images.githubusercontent.com/24280391/148690135-9ff67950-5ae8-4fe0-a5e7-fe8097d4f428.png)

## Installation

```
pip install -r requirements.txt
```

Install PyTorch3D from: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md


## Usage

- Download ShapeNet dataset and store it in **data** folder
- Reduce the meshes in ShapeNet to at most 100 vertices using the notebook **data/Generation Notebook.ipynb**
- Train the vertix prediction model using the following command: 
  ```
  python train_vertix.py --config=configs/vertix_train.json
  ```
- Train the edge prediction model using the following command: 
  ```
  python train_edge.py --config=configs/edge_train.json
  ```
- Train the face prediction model using the following command: 
  ```
  python train_face.py --config=configs/face_train.json
  ```
- Use the visualization notebook **Visualize Results.ipynb** to visualize the trained model on ShapeNet meshes.

## System Results


![visualizations](visualizations.png)


