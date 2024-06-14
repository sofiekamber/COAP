import os
import pathlib
import argparse
import collections

import yaml
import torch
import smplx
import trimesh
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from skimage.io import imsave

from coap import attach_coap

from data import SMPLDataset
from halo.halo.models.halo_adapter.adapter import HaloAdapter
from halo.halo.models.halo_adapter.converter import transform_to_canonical
from  halo.halo.models.halo_adapter.converter import PoseConverter
from halo.halo.models.halo_adapter.interface import convert_joints, change_axes
from renderer import Renderer

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def visualize_point_cloud(red=None, blue=None, green=None, mesh=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if red is not None:
        ax.scatter(red[:, 0], red[:, 1], red[:, 2], c='r')
        for i in range(len(red)):  # plot each point + it's index as text above
            ax.text(red[i, 0], red[i, 1], red[i, 2], '%s' % (str(i)), size=20, zorder=1,
                    color='k')
    if blue is not None:
        ax.scatter(blue[:, 0], blue[:, 1], blue[:, 2], c='b')
        for i in range(len(blue)):  # plot each point + it's index as text above
            ax.text(blue[i, 0], blue[i, 1], blue[i, 2], '%s' % (str(i)), size=20, zorder=1,
                    color='k')
    if green is not None:
        ax.scatter(green[:, 0], green[:, 1], green[:, 2], c='g')
    if mesh is not None:
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show(block=True)


class COAPModule(pl.LightningModule):
    def __init__(self, smpl_cfg, data_cfg, train_cfg, args):
        super().__init__()
        self.smpl_cfg = smpl_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.args = args

        smpl_body = smplx.create(**smpl_cfg, batch_size=train_cfg['batch_size'])
        self.smpl_body = attach_coap(smpl_body, pretrained=False)
        self.renderer = Renderer()

        self.max_queries = 100000  # to prevent OOM at inference time

    def configure_optimizers(self):
        return torch.optim.Adam(self.smpl_body.coap.parameters(), lr=1e-4)

    def state_dict(self, *args, **kwargs):
        return self.smpl_body.coap.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.smpl_body.coap.load_state_dict(*args, **kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(SMPLDataset.from_config(self.smpl_cfg, self.data_cfg, split='train'), drop_last=True,
            num_workers=self.train_cfg['num_workers'], batch_size=self.train_cfg['batch_size'], pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(SMPLDataset.from_config(self.smpl_cfg, self.data_cfg, split='val'), drop_last=True,
            num_workers=self.train_cfg['num_workers'], batch_size=self.train_cfg['batch_size'], pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()

    def forward(self, batch, points, ret_intermediate=False):
        smpl_output = self.smpl_body(**batch, return_verts=True, return_full_pose=True)
        return self.smpl_body.coap.query(points, smpl_output, ret_intermediate=ret_intermediate)

    def training_step(self, batch, batch_idx):
        pred_occ, inter_log = self(batch, batch['points'], ret_intermediate=True)
        tr_inds = ~inter_log['all_out']
        loss = F.mse_loss(pred_occ[tr_inds], batch['gt_occ'][tr_inds])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.trainer.is_global_zero: # extract and render meshes for the first batch only
            smpl_output = self.smpl_body(**batch, return_verts=True, return_full_pose=True)
            meshes, images, smpl_images = self.visualize(batch, smpl_output)
            image = torch.from_numpy(np.concatenate(images[:6], axis=1)).permute(2, 0, 1)  # CHW
            smpl_images = torch.from_numpy(np.concatenate(smpl_images[:6], axis=1)).permute(2, 0, 1)
            image = torch.cat((image, smpl_images), dim=1)
            self.logger.experiment.add_image(f'val/renderings', image, self.global_step)

        val_metric = self.compute_val_loss(batch)
        for key, val in val_metric.items():
            self.log(f"val_{key}", val, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        val_metric = self.compute_val_loss(batch)
        for key, val in val_metric.items():
            self.log(f"test_{key}", val, on_step=True, on_epoch=True, sync_dist=True)

        if self.args.eval_export_visuals:
            self.export_visuals(batch, batch['seq_names'], batch['frame_ids'])

        if self.args.eval_export_halo:
            self.export_halo(batch, batch['seq_names'], batch['frame_ids'])
        return val_metric

    def compute_val_loss(self, batch):
        points = batch['points']
        gt_occ = batch['gt_occ']
        batch_size, n_pts, _ = points.shape

        # prevent OOM
        pred_occ = []
        for pts in torch.split(points, self.max_queries//batch_size, dim=1):
            pred_occ.append(self(batch, pts))

        pred_occ_halo = []
        for pts in torch.split(points, self.max_queries//batch_size, dim=1):
            pred_occ_halo.append(self.query_halo(batch, pts, is_right=self.smpl_cfg['is_rhand']))

        pred_occ = torch.cat(pred_occ, dim=1)
        iou_unif = self.smpl_body.coap.compute_iou(pred_occ[:, :n_pts//2], gt_occ[:, :n_pts//2])
        iou_surf = self.smpl_body.coap.compute_iou(pred_occ[:, n_pts//2:], gt_occ[:, n_pts//2:])
        iou_mean = (iou_unif + iou_surf)*0.5

        pred_occ_halo = torch.cat(pred_occ_halo, dim=1)
        iou_unif_halo = self.smpl_body.coap.compute_iou(pred_occ_halo[:, :n_pts//2], gt_occ[:, :n_pts//2])
        iou_surf_halo = self.smpl_body.coap.compute_iou(pred_occ_halo[:, n_pts//2:], gt_occ[:, n_pts//2:])
        iou_mean_halo = (iou_unif_halo + iou_surf_halo)*0.5
        return dict(iou_unif=iou_unif, iou_surf=iou_surf, iou_mean=iou_mean, iou_unif_halo=iou_unif_halo, iou_surf_halo=iou_surf_halo, iou_mean_halo=iou_mean_halo)

    def test_epoch_end(self, outputs):
        """ Aggregate test predictions.

        Args:
            outputs (list): list of dictionaries containing scores
        """
        test_metric = outputs[0].keys()
        val_metric = {key: torch.stack([x[key] for x in outputs]) for key in test_metric}
        agg_metric = {}
        for key in test_metric:
            agg_metric[key] = torch.mean(self.all_gather(val_metric[key]))

        if self.trainer.is_global_zero:  # to avoid deadlock
            for key, val in agg_metric.items():
                self.log(f"test_end_{key}", val, rank_zero_only=True)
            val_metric = {key: float(val.item() if torch.is_tensor(val) else val) for key, val in agg_metric.items()}

            dst_path = os.path.join(self.logger.save_dir, f'test_epoch={self.current_epoch:05d}_step={self.global_step:05d}.yml')
            with open(dst_path, 'w') as f:
                yaml.dump(val_metric, f)

            print('\n\nTest results are saved in:', dst_path, end='\n\n')

    def export_halo(self, batch, seq_names:list, frame_ids:list):
        smpl_output = self.smpl_body(**batch, return_verts=True, return_full_pose=True)

        coap_meshes = self.smpl_body.coap.extract_mesh(smpl_output, max_queries=self.max_queries, use_mise=True)
        #
        coap_meshes = [coap_meshes[0]]  # only first batch
        batch_size = smpl_output.vertices.shape[0]

        meshes = self.visualize_halo(batch, smpl_output, smpl_cfg['is_rhand'])

        for b_ind, mesh in enumerate(zip(meshes)):
            seq_name = seq_names[b_ind]
            frame_id = frame_ids[b_ind]

            dst_mesh_dir = os.path.join(self.logger.save_dir, 'halo_meshes', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
            pathlib.Path(dst_mesh_dir).mkdir(parents=True, exist_ok=True)

            # TODO remove 0
            mesh[0].export(os.path.join(dst_mesh_dir, f'{frame_id}.ply'))

            coap_meshes[b_ind].export(os.path.join(dst_mesh_dir, f'{frame_id}_COAP_MESH.ply'))

            smpl_output_mesh = trimesh.Trimesh(smpl_output.vertices[b_ind].detach().cpu().numpy(), self.smpl_body.faces)
            smpl_output_mesh.export(os.path.join(dst_mesh_dir, f'{frame_id}_MANO_MESH.ply'))

    def export_visuals(self, batch, seq_names:list, frame_ids:list):
        smpl_output = self.smpl_body(**batch, return_verts=True, return_full_pose=True)
        batch_size = smpl_output.vertices.shape[0]

        meshes, images, smpl_images = self.visualize(batch, smpl_output)
        for b_ind, (mesh, image, smpl_image) in enumerate(zip(meshes, images, smpl_images)):
            seq_name = seq_names[b_ind]
            frame_id = frame_ids[b_ind]

            dst_mesh_dir = os.path.join(self.logger.save_dir, 'meshes', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
            dst_image_dir = os.path.join(self.logger.save_dir, 'images', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
            dst_smpl_image_dir = os.path.join(self.logger.save_dir, 'smpl_images', f'epoch={self.current_epoch:05d}_step={self.global_step:05d}', seq_name)
            pathlib.Path(dst_mesh_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(dst_image_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(dst_smpl_image_dir).mkdir(parents=True, exist_ok=True)

            mesh.export(os.path.join(dst_mesh_dir, f'{frame_id}.ply'))
            imsave(os.path.join(dst_image_dir, f'{frame_id}.png'), image)
            imsave(os.path.join(dst_smpl_image_dir, f'{frame_id}.png'), smpl_image)

            global_orient_init = smplx.lbs.batch_rodrigues(batch['global_orient_init'].reshape(-1, 3))
            global_orient_init = global_orient_init.cpu().numpy()

            mesh_vertices = smpl_output.vertices[b_ind].detach().cpu().numpy()
            mesh_vertices = (global_orient_init[b_ind].T @ mesh_vertices.T).T

            # Export GT Mesh
            posed_mesh = trimesh.Trimesh(mesh_vertices, self.smpl_body.faces)

            # Save mesh
            posed_mesh.export(os.path.join(dst_mesh_dir, f'{frame_id}_MANO_MESH.ply'))

    @torch.no_grad()
    def visualize(self, batch, smpl_output):
        meshes = self.smpl_body.coap.extract_mesh(smpl_output, max_queries=self.max_queries, use_mise=True)
        global_orient_init = None
        smpl_vertices = smpl_output.vertices.clone()
        if 'global_orient_init' in batch:  # normalize visualization wrt the first frame
            global_orient_init = smplx.lbs.batch_rodrigues(batch['global_orient_init'].reshape(-1, 3))  # B,3,3
            smpl_vertices = (global_orient_init.transpose(1, 2) @ smpl_vertices.transpose(1, 2)).transpose(1, 2)
            global_orient_init = global_orient_init.cpu().numpy()
            for i, mesh in enumerate(meshes):
                mesh.vertices = (global_orient_init[i].T @ mesh.vertices.T).T

        rnd_images, rnd_smpl_images = [], []
        for i, mesh in enumerate(meshes):
            # save image
            rnd_images.append((self.renderer.render_mesh(
                torch.tensor(mesh.vertices).float().unsqueeze(0).to(self.device),
                torch.tensor(mesh.faces).unsqueeze(0).to(self.device),
                torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255,
                mode='t'
            )[0].squeeze(0) * 255.).cpu().numpy().astype(np.uint8))
            rnd_smpl_images.append((self.renderer.render_mesh(
                smpl_vertices[i].float().unsqueeze(0).to(self.device),
                self.smpl_body.faces_tensor.unsqueeze(0).to(self.device),
                mode='p'
            )[0].squeeze(0) * 255.).cpu().numpy().astype(np.uint8))
        return meshes, rnd_images, rnd_smpl_images

    def query_halo(self, batch, points, is_right=True):
        smpl_output = self.smpl_body(**batch, return_verts=True, return_full_pose=True)
        halo_config_file = 'configs/halo/halo_config.yaml'
        halo_adapter = HaloAdapter(halo_config_file, device='cuda:0', denoiser_pth=None)
        target_js = convert_joints(smpl_output.joints, source='halo', target='mano')

        # flip points and joints for left hand
        if not is_right:
            flip_tensor = torch.tensor([-1, 1, 1], device='cuda:0')

            target_js -= target_js[:, 0:1, :]
            target_js[:, :, 0] *= flip_tensor[0]

            points -= target_js[:, 0:1, :]
            points[:, :, 0] *= flip_tensor[0]

            print("target_js", target_js.shape)
            print("points", points.shape)

        occ = halo_adapter.query_points(points * 100, target_js.cuda() * 100, joint_order="mano")
        return occ

    @torch.no_grad()
    def visualize_halo(self, batch, smpl_output, is_right=True):
        device = smpl_output.vertices.device
        halo_config_file = 'configs/halo/halo_config.yaml'
        halo_adapter = HaloAdapter(halo_config_file, device='cuda:0', denoiser_pth=None)
        output_joints = smpl_output.joints

        if not is_right:
            flip_tensor = torch.tensor([-1, 1, 1], device='cuda:0')
            output_joints[:, :, :] *= flip_tensor

        first_batch = output_joints[0].unsqueeze(0)

        target_js = convert_joints(first_batch, source='halo', target='mano')

        halo_mesh, halo_kpts = halo_adapter(target_js.cuda() * 100, original_position=True, joint_order="mano", return_kps=True)
        halo_mesh.vertices = halo_mesh.vertices / 100

        # TODO not only first batch (remove list
        return [halo_mesh]


if __name__ == "__main__":
    # add input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to a training configuration file.')
    parser.add_argument('--out_dir', required=True, type=str, help='Log directory.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the SMPL body models.')
    parser.add_argument('--data_root', required=True, type=str, help='AMASS data root.')
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path/URL of the checkpoint from which training is resumed.')
    parser.add_argument('--run_eval', action='store_true', help='Whether to run evaluation instead of training.')
    parser.add_argument('--eval_export_visuals', action='store_true', help='Whether to export meshes and render images in the evaluation mode')
    parser.add_argument('--eval_export_halo', action='store_true', help='Whether to export HALO meshes in the evaluation mode')
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    smpl_cfg, data_cfg, train_cfg = cfg['smpl_cfg'], cfg['data_cfg'], cfg['train_cfg']
    smpl_cfg['model_path'] = args.model_path
    data_cfg['data_root'] = args.data_root

    # create trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f'{args.out_dir}/ckpts/',
        filename='model-{epoch:04d}-{val_iou_mean:.6f}', every_n_epochs=args.checkpoint_every_n_epochs,
        save_top_k=-1, save_last=True, verbose=True,
    )

    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[checkpoint_callback, pl.callbacks.TQDMProgressBar(refresh_rate=10)],
        logger=pl.loggers.TensorBoardLogger(args.out_dir, os.path.basename(args.out_dir)),
    )

    # path to resume training
    if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
        ckpt_path = args.ckpt_path
    elif os.path.exists(os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")):
        ckpt_path = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
    else:
        ckpt_path = None

    # create model
    model = COAPModule(smpl_cfg, data_cfg, train_cfg, args)
    if args.run_eval:
        trainer.test(model, ckpt_path=ckpt_path, verbose=True)
    else:
        trainer.fit(model, ckpt_path=ckpt_path)
