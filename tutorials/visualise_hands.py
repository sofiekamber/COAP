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
        #posed_mesh = trimesh.Trimesh(vertices=posed_mesh.vertices, faces=posed_mesh.faces)
        posed_mesh = trimesh.Trimesh(smpl_output.vertices[0].detach().cpu().numpy(), model.faces)

        VIEWER.scene.add(pyrender.Mesh.from_trimesh(posed_mesh))

    #VIEWER.scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
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

#m = [15, 14, 13, 3, 2, 1, 6, 5, 4, 12, 11, 10, 9, 8, 7, 0]
m = range(0, 16)
#m = [0,4,5,6,1,2,3,7,8,9,10,11,12,13,14,15]
def reorder(pose):
    res = []
    for i in range(16):
        if True:
            res.append(pose[0][3*m[i]])
            res.append(pose[0][3*m[i]+1])
            res.append(pose[0][3*m[i]+2])
        else:
            res.append(0)
            res.append(0)
            res.append(0)
    return np.array([res])
def main():
    i = 0
    side = 'left'
    with open("samples/selfpen_examples/mano/all_train.json", 'r') as f:
        all = json.load(f)
    for x in all.values():
        for k,j in x.items():
            #print(k)
            if j[side]:
                if i == 100:
                    print(k)
                    with open("./samples/selfpen_examples/mano/hand.pkl", 'wb') as f_p:
                        pickle.dump(j, f_p)
                    return

                mano_pose = torch.FloatTensor(np.array([j[side]['pose']])).view(-1, 3).to('cuda')
                root_pose = mano_pose[0].view(1, 3).to('cuda')
                hand_pose = mano_pose[1:, :].view(1, -1).to('cuda')
                shape = torch.FloatTensor(np.array([j[side]['shape']])).view(1, -1).to('cuda')
                trans = torch.FloatTensor(np.array(j[side]['trans'])).view(1, 3).to('cuda')

                data = dict()
                data['global_orient'] = root_pose
                data['hand_pose'] = hand_pose
                data['shape'] = shape
                data['transl'] = trans

                scene_mesh = trimesh.load_mesh('/home/rafael/PycharmProjects/DigitalHumans/volumetric-hand-model/COAP-hand/tutorials/samples/scene_collision/bunny.obj')
                model = smplx.create(model_path="/home/rafael/Downloads/Models", is_rhand=side == 'right', model_type='mano',
                                     use_pca=False).to('cuda')
                model = attach_coap(model, pretrained=True, device='cuda')

                # visualize
                smpl_output = model(**data)
                #smpl_output = model(global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                # NOTE: make sure that smpl_output contains the valid SMPL variables (pose parameters, joints, and vertices).
                assert model.joint_mapper is None, 'COAP requires valid SMPL joints as input'
                #print(data)
                visualize(model, smpl_output, scene_mesh)
                i += 1


VISUALIZE = True
if VISUALIZE:
    VIEWER = pyrender.Viewer(
        pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0]),
        use_raymond_lighting=True, run_in_thread=True, viewport_size=(1920, 1080))

main()