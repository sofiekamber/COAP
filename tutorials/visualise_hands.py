import os
import sys
import time
import pickle
import pathlib
import argparse
import copy

import smplx
import torch
import trimesh
import pyrender
import numpy as np
import json
import torch.nn.functional as F
from pytorch3d import transforms

from coap import attach_coap

@torch.no_grad()
def visualize(model=None, smpl_output=None, scene_mesh=None, query_samples=None, collision_samples=None):
    if not VISUALIZE:
        return

    def vis_create_pc(pts, color=(0.0, 1.0, 0.0), radius=0.005):
        if torch.is_tensor(pts):
            pts = pts.cpu().numpy()

        tfs = np.tile(np.eye(4), (pts.shape[0], 1, 1))
        tfs[:, :3, 3] = pts
        sm_in = trimesh.creation.uv_sphere(radius=radius)
        sm_in.visual.vertex_colors = color

        return pyrender.Mesh.from_trimesh(sm_in, poses=tfs)

    VIEWER.render_lock.acquire()
    # clear scene
    while len(VIEWER.scene.mesh_nodes) > 0:
        VIEWER.scene.mesh_nodes.pop()

    if smpl_output is not None:
        #posed_mesh = model.coap.extract_mesh(smpl_output, use_mise=True)[0]
        # posed_mesh = trimesh.Trimesh(vertices=posed_mesh.vertices, faces=posed_mesh.faces)
        posed_mesh = trimesh.Trimesh(smpl_output.vertices[0].detach().cpu().numpy(), model.faces)

        VIEWER.scene.add(pyrender.Mesh.from_trimesh(posed_mesh))

    VIEWER.scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
    if query_samples is not None:
        VIEWER.scene.add(vis_create_pc(query_samples, color=(0.0, 1.0, 0.0)))
    if collision_samples is not None:
        VIEWER.scene.add(vis_create_pc(collision_samples, color=(1.0, 0.0, 0.0)))

    VIEWER.render_lock.release()

def to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device=device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device)
    return x

def main():
    side = 'right'
    with open("samples/selfpen_examples/mano/all_val.json", 'r') as f:
        all = json.load(f)
    for x in all.values():
        for j in x.values():
            if j[side]:
                j[side]['pose'] = np.array([j[side]['pose']], dtype=np.float32)
                j[side]['shape'] = np.array([j[side]['shape']], dtype=np.float32)
                j[side]['trans'] = np.array([j[side]['trans']], dtype=np.float32)

                torch_param = {key: to_tensor(val, 'cuda') for key, val in j.items()}
                #print(torch_param)
                smpl_body_pose = torch.zeros((1, 48), dtype=torch.float, device='cuda')
                smpl_body_pose[:, :48] = torch.from_numpy(torch_param['right' if torch_param['right'] else 'left']['pose']).to('cuda')
                torch_param['hand_pose'] = smpl_body_pose.to(torch.float32)
                torch_param['betas'] = torch.from_numpy(torch_param['right' if torch_param['right'] else 'left']['shape']).to(torch.float32).to('cuda')
                torch_param['transl'] = torch.from_numpy(torch_param['right' if torch_param['right'] else 'left']['trans']).to(torch.float32).to('cuda')

                data = torch_param
                scene_mesh = trimesh.load_mesh('/home/rafael/PycharmProjects/DigitalHumans/volumetric-hand-model/COAP-hand/tutorials/samples/scene_collision/bunny.obj')
                key_pose = 'hand_pose'
                print(data[key_pose])
                model = smplx.create(model_path="/home/rafael/Downloads/Models", is_rhand=bool(data['right']), model_type='mano', gender='neutral',
                                     num_pca_comps=1)
                model = attach_coap(model, pretrained=True, device='cuda')

                scene_vertices = torch.from_numpy(scene_mesh.vertices).to(device='cuda', dtype=torch.float)
                scene_normals = torch.from_numpy(np.asarray(scene_mesh.vertex_normals).copy()).to(device='cuda',
                                                                                                  dtype=torch.float)

                # visualize
                smpl_output = model(**data, return_verts=True, return_full_pose=True)
                # NOTE: make sure that smpl_output contains the valid SMPL variables (pose parameters, joints, and vertices).
                assert model.joint_mapper is None, 'COAP requires valid SMPL joints as input'
                #print(data)
                visualize(model, smpl_output, scene_mesh)

VISUALIZE = True
if VISUALIZE:
    VIEWER = pyrender.Viewer(
        pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0]),
        use_raymond_lighting=True, run_in_thread=True, viewport_size=(1920, 1080))

main()