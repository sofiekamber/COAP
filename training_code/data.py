import os
import glob
import itertools
import collections

import torch
import smplx
import trimesh
import numpy as np
import torch.nn.functional as F
import json

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from leap.tools.libmesh import check_mesh_contains

from coap import attach_coap

SEAL_FACES_R = [
    [120, 108, 778],
    [108, 79, 778],
    [79, 78, 778],
    [78, 121, 778],
    [121, 214, 778],
    [214, 215, 778],
    [215, 279, 778],
    [279, 239, 778],
    [239, 234, 778],
    [234, 92, 778],
    [92, 38, 778],
    [38, 122, 778],
    [122, 118, 778],
    [118, 117, 778],
    [117, 119, 778],
    [119, 120, 778],
]
# vertex ids around the ring of the wrist
CIRCLE_V_ID = np.array(
    [108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120],
    dtype=np.int64,
)


def seal_mano_mesh(v3d, faces, is_rhand, device):
    # v3d: B, 778, 3
    # faces: 1538, 3
    # output: v3d(B, 779, 3); faces (1554, 3)

    seal_faces = torch.LongTensor(np.array(SEAL_FACES_R)).to(device)
    if not is_rhand:
        # left hand
        seal_faces = seal_faces[:, np.array([1, 0, 2])]  # invert face normal
    centers = v3d[:, CIRCLE_V_ID].mean(dim=1)[:, None, :]
    sealed_vertices = torch.cat((v3d, centers), dim=1)
    faces = torch.from_numpy(faces.astype(np.int64)).to(device)
    faces = torch.cat((faces, seal_faces), dim=0)
    return sealed_vertices, faces


class SMPLDataset(torch.utils.data.Dataset):

    @torch.no_grad()
    def __init__(self, smpl_cfg, amass_data, **data_cfg):
        super().__init__()
        self.smpl_cfg = smpl_cfg
        self.amass_data = amass_data

        self.data_keys = list(amass_data.keys())
        self.smplx_body = attach_coap(smplx.create(**smpl_cfg), pretrained=False)
        self.faces = self.smplx_body.faces.copy()

        self.points_sigma = 0.01
        self.points_uniform_ratio = 0.5
        self.n_points = data_cfg.get('n_points', 256)

    @classmethod
    @torch.no_grad()
    def from_config(cls, smpl_cfg, data_cfg, split='train'):
        # load data
        gender = smpl_cfg['gender']
        smpl_body = smplx.create(**smpl_cfg)
        model_type = smpl_cfg['model_type']

        if model_type == 'mano':
            num_body_joints = smpl_body.NUM_HAND_JOINTS
        else:
            num_body_joints = smpl_body.NUM_BODY_JOINTS

        data_root = data_cfg['data_root']
        datasets = data_cfg[split]['datasets']
        select_every = data_cfg[split].get('select_every', 1)

        dataset = collections.defaultdict(list)
        if smpl_cfg['model_type'] == 'mano':
            hand_side = 'right' if smpl_body.is_rhand else 'left'
            print(f"mano {hand_side}")

            for ds in datasets:
                f = open(os.path.join(data_root, ds, split + '.json'))
                data = json.load(f)
                for capture_id, frames in data.items():
                    seq_id = 0
                    seq_name = capture_id + "_" + str(seq_id)
                    betas_seq = []
                    body_pose_seq = []
                    global_orient_seq = []
                    seq_names_seq = []
                    global_orient_init_seq = []
                    global_orient_init_3D = []
                    transl_seq = []
                    frame_ids_seq = []
                    last_frame_id = 0
                    for (frame, hand_params) in frames.items():
                        # Check if new sequence
                        if not hand_params[hand_side]:
                            # If hand side is not available, skip
                            continue
                        if int(frame) % select_every == 0:
                            if last_frame_id + select_every != int(frame) and seq_names_seq:
                                if len(frame_ids_seq) > 1:
                                    # Store other sequence
                                    dataset['betas'].append(np.array(betas_seq))
                                    dataset['hand_pose'].append(np.array(body_pose_seq))
                                    dataset['global_orient'].append(np.array(global_orient_seq))
                                    dataset['global_orient_init'].append(np.array(global_orient_init_seq))
                                    dataset['transl'].append(np.array(transl_seq))
                                    dataset['seq_names'].append(seq_names_seq)
                                    dataset['frame_ids'].append(frame_ids_seq)

                                    # Create new sequence
                                    seq_id += 1
                                    seq_name = capture_id + "_" + str(seq_id)

                                betas_seq = []
                                body_pose_seq = []
                                global_orient_seq = []
                                seq_names_seq = []
                                global_orient_init_seq = []
                                global_orient_init_3D = []
                                transl_seq = []
                                frame_ids_seq = []

                            pose = hand_params[hand_side]['pose']
                            betas_seq.append(hand_params[hand_side]['shape'][:smpl_body.num_betas])
                            transl_seq.append(hand_params[hand_side]['trans'])

                            global_orient_seq.append(pose[:3])
                            body_pose_seq.append(pose[3:3 + num_body_joints * 3])
                            seq_names_seq.append(seq_name)
                            frame_ids_seq.append(frame)

                            if not global_orient_init_3D:
                                global_orient_init_3D = pose[:3]
                            global_orient_init_seq.append(global_orient_init_3D)
                            last_frame_id = int(frame)

                    # Save last sequence
                    dataset['betas'].append(np.array(betas_seq))
                    dataset['hand_pose'].append(np.array(body_pose_seq))
                    dataset['global_orient'].append(np.array(global_orient_seq))
                    dataset['global_orient_init'].append(np.array(global_orient_init_seq))
                    dataset['transl'].append(np.array(transl_seq))
                    dataset['seq_names'].append(seq_names_seq)
                    dataset['frame_ids'].append(frame_ids_seq)

        else:
            for ds in datasets:
                subject_dirs = [s_dir for s_dir in sorted(glob.glob(os.path.join(data_root, ds, '*'))) if
                                os.path.isdir(s_dir)]
                for subject_dir in subject_dirs:
                    seq_paths = [sn for sn in glob.glob(os.path.join(subject_dir, '*.npz')) if
                                 not sn.endswith('shape.npz') and not sn.endswith('neutral_stagei.npz')]
                    for seq_path in seq_paths:
                        seq_sample = np.load(seq_path, allow_pickle=True)
                        pose = seq_sample['poses'][::select_every]
                        betas = seq_sample['betas'].reshape((1, -1))[:, :smpl_body.num_betas]
                        n_frames = pose.shape[0]

                        dataset['betas'].append(betas.repeat(n_frames, axis=0))
                        dataset['global_orient'].append(pose[:, :3])
                        dataset['body_pose'].append(pose[:, 3:3 + num_body_joints * 3])

                        dataset['global_orient_init'].append(pose[:1, :3].repeat(n_frames, axis=0))
                        seq_name = os.path.join(os.path.basename(subject_dir),
                                                os.path.splitext(os.path.basename(seq_path))[0])
                        dataset['seq_names'].append([seq_name] * n_frames)
                        dataset['frame_ids'].append(
                            list(map(lambda x: f'{x:06d}', list(range(seq_sample['poses'].shape[0]))[::select_every])))

                        if model_type == 'smplx' or model_type == 'smplh':
                            b_ind = 3 + num_body_joints * 3
                            dataset['left_hand_pose'].append(pose[:, b_ind:b_ind + 45])
                            dataset['right_hand_pose'].append(pose[:, b_ind + 45:b_ind + 2 * 45])
                        elif model_type == 'smpl':  # flatten hands for smpl
                            dataset['body_pose'][-1][:, -6:] = 0

        data = {}
        for key, val in dataset.items():
            if isinstance(val[0], np.ndarray):
                data[key] = torch.from_numpy(np.concatenate(val, axis=0)).float()
            else:
                data[key] = list(itertools.chain.from_iterable(val))

        return SMPLDataset(smpl_cfg, data)

    def sample_points(self, smpl_output):
        bone_trans = self.smplx_body.coap.compute_bone_trans(smpl_output.full_pose, smpl_output.joints)
        bbox_min, bbox_max = self.smplx_body.coap.get_bbox_bounds(smpl_output.vertices,
                                                                  bone_trans)  # (B, K, 1, 3) [can space]
        n_parts = bbox_max.shape[1]

        #### Sample points inside local boxes
        n_points_uniform = int(self.n_points * self.points_uniform_ratio)
        n_points_surface = self.n_points - n_points_uniform

        bbox_size = (bbox_max - bbox_min).abs() * self.smplx_body.coap.bbox_padding - 1e-3  # (B,K,1,3)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bb_min = (bbox_center - bbox_size * 0.5)  # to account for padding

        uniform_points = bb_min + torch.rand((1, n_parts, n_points_uniform, 3)) * bbox_size  # [0,bs] (B,K,N,3)

        # project points to the posed space
        abs_transforms = torch.inverse(bone_trans)  # B,K,4,4
        uniform_points = (abs_transforms.reshape(1, n_parts, 1, 4, 4).repeat(1, 1, n_points_uniform, 1, 1) @ F.pad(
            uniform_points, [0, 1], "constant", 1.0).unsqueeze(-1))[..., :3, 0]

        #### Sample surface points
        meshes = Meshes(smpl_output.vertices.float().expand(n_parts, -1, -1),
                        self.smplx_body.coap.get_tight_face_tensor())
        surface_points = sample_points_from_meshes(meshes, num_samples=n_points_surface)
        surface_points += torch.from_numpy(np.random.normal(scale=self.points_sigma, size=surface_points.shape))
        surface_points = surface_points.reshape((1, n_parts, -1, 3))

        points = torch.cat((uniform_points, surface_points), dim=-2).float()  # B,K,n_points,3

        if self.smpl_cfg['model_type'] == 'mano':
            vertices, faces = seal_mano_mesh(smpl_output.vertices, self.smplx_body.faces, self.smplx_body.is_rhand, smpl_output.vertices.device)
        else:
            vertices = smpl_output.vertices
            faces = self.faces

        #### Check occupancy
        points = points.reshape(-1, 3).numpy()
        mesh = trimesh.Trimesh(vertices.squeeze().numpy(), faces, process=False)
        gt_occ = check_mesh_contains(mesh, points).astype(np.float32)

        return dict(points=points, gt_occ=gt_occ)

    @torch.no_grad()
    def __getitem__(self, idx):
        smpl_data = {key: self.amass_data[key][idx:idx + 1] for key in self.data_keys}
        smpl_output = self.smplx_body(**smpl_data, return_verts=True, return_full_pose=True)  # smpl fwd pass
        smpl_data = {key: val.squeeze(0) if torch.is_tensor(val) else val[0] for key, val in
                     smpl_data.items()}  # remove B dim
        smpl_data.update(self.sample_points(smpl_output))
        return smpl_data

    def __len__(self):
        return len(self.amass_data[self.data_keys[0]])
