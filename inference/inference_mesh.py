import torch

from data.shapenet import ShapeNet
from model.vertix_edge_model import VertixEdge
from model.face_model import FaceModel
from inference.inference_vertix_edge import InferenceHandlerVertixEdgeModel
from inference.inference_face_model import InferenceHandlerFaceModel
import numpy as np
import trimesh


class InferenceHandlerMesh:

    def __init__(self, vertix_edge_ckpt, face_model_ckpt, num_vertices, feature_size, device):

        self.device=device
        self.num_vertices = num_vertices
        self.vertix_edge_inferer = InferenceHandlerVertixEdgeModel(vertix_edge_ckpt, num_vertices, feature_size, device)
        self.face_model_inferer = InferenceHandlerFaceModel(face_model_ckpt, num_vertices, feature_size, device)

    def infer_single(self, input_sdf, mask, v1s_idx, v2s_idx, adj):


        vertices, edges = self.vertix_edge_inferer.infer_single(input_sdf, mask, v1s_idx,v2s_idx,adj)
        
        candidate_faces = []

        for i in range(self.num_vertices):
            for j in range(i+1,self.num_vertices):
                for k in range(j+1,self.num_vertices):
                    if edges[i][j] and edges[j][k] and edges[k][i]:     
                        candidate_faces.append(np.array([i,j,k]).reshape(1,-1))

        candidate_faces = np.concatenate(candidate_faces,0)

        edge_idx = trimesh.graph.face_adjacency(candidate_faces)

        hv = np.zeros((len(candidate_faces),7))

        for i in range(len(candidate_faces)):


            
            curr_face = candidate_faces[i]

            v1 = vertices[curr_face[0]]

            v2 = vertices[curr_face[1]]

            v3 = vertices[curr_face[2]]

            centroid = (v1+v2+v3) / 3

            radius = np.linalg.norm(v2 - v1) + np.linalg.norm(v3 - v2) + np.linalg.norm(v3 - v1)

            normal = np.cross(v2 - v1, v3-v2)

            hv[i] = np.array([centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], radius])



        face_mask = self.face_model_inferer.infer_single(hv, edge_idx.transpose(1,0))

        output_faces = candidate_faces[face_mask == 1]


        return vertices, candidate_faces, output_faces
    


        
