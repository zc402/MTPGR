"""
The original demo.py in VIBE has a bug with GRU usage:
    It reads a batch (450 frames) and run GRU, then reads another batch, with previous GRU memory abandoned.
    This leads to unsmooth results.
"""
import os
from typing import List

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from vibe.models.vibe import VIBE_Demo
from vibe.utils.renderer import Renderer
from vibe.dataset.inference import Inference
from vibe.utils.smooth_pose import smooth_pose
from vibe.data_utils.kp_utils import convert_kps
from vibe.utils.pose_tracker import run_posetracker

from vibe.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)


class Video2Smpl:
    """
    video -> trace -> police trace (manually) -> smpl parameters
    The trace of the police needs to be distinguished from other traces. for this purpose, the longest trace is preserved.
    """
    def __init__(self):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.display_trace = False
        self.display_render = False

    def __call__(self, video_file):
        image_folder, num_frames, img_shape = self._video2images(video_file)
        trace = self._images2trace(image_folder)
        smpl_folder = self._make_smpl_folder(video_file)
        vibe_results = self._trace2smpl(image_folder, img_shape, trace, smpl_folder)
        if self.display_render:
            self._render(image_folder, img_shape, num_frames, vibe_results)
        self._rm_tmp(image_folder)

    @staticmethod
    def _video2images(video_file: Path):
        # Step1: video to image sequence, save to /tmp

        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')
        video_file = str(video_file)
        image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
        print(f'Number of frames {num_frames}')
        return image_folder, num_frames, img_shape

    def _images2trace(self, image_folder):

        # Run tracker
        mot = MPT(
            device=self.device,
            batch_size=2,  # original: 12
            display=self.display_trace,
            detector_type='yolo',
            output_format='dict',
            yolo_img_size=416,
        )
        tracking_results = mot(image_folder)

        # Preserve longest trace
        trace_len: List = [tracking_results[person_id]['frames'].shape[0]
                           for person_id in list(tracking_results.keys())]
        longest_trace = tracking_results[list(tracking_results.keys())[np.argmax(trace_len)]]
        longest_trace = {0: longest_trace}  # assign to person_id: 0
        return longest_trace

    @staticmethod
    def _make_smpl_folder(video_file: Path):
        output_folder_base = video_file.parent  # Dataset folder
        output_folder = os.path.join(output_folder_base, os.path.basename(video_file)+".smpl")  # .../001.mp4.smpl as folder
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def _trace2smpl(self, image_folder, image_shape, tracking_results, output_folder):

        tracking_method = 'bbox'
        vibe_batch_size = 1  # original: 200

        # ========= Define VIBE model ========= #
        model = VIBE_Demo(
            seqlen=16,  # original: 16
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(self.device)

        # ========= Load pretrained weights ========= #
        pretrained_file = download_ckpt(use_3dpw=False)
        ckpt = torch.load(pretrained_file)
        ckpt = ckpt['gen_state_dict']
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        print(f'Loaded pretrained VIBE weights from \"{pretrained_file}\"')

        # ========= demo.py from VIBE ========= #
        vibe_results = {}
        for person_id in list(tracking_results.keys()):
            bboxes = joints2d = None

            if tracking_method == 'bbox':
                bboxes = tracking_results[person_id]['bbox']
            elif tracking_method == 'pose':
                joints2d = tracking_results[person_id]['joints2d']

            frames = tracking_results[person_id]['frames']

            bbox_scale = 1.1
            dataset = Inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                scale=bbox_scale,
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            has_keypoints = True if joints2d is not None else False

            dataloader = DataLoader(dataset, batch_size=vibe_batch_size, num_workers=16)

            with torch.no_grad():

                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    batch = batch.unsqueeze(0)
                    batch = batch.to(self.device)

                    batch_size, seqlen = batch.shape[:2]
                    output = model(batch)[-1]

                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:, :, 3:75].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, :, 75:].reshape(batch_size * seqlen, -1))
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                    smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))

                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)
                smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
                del batch

            # ========= Save results to a pickle file ========= #
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()
            smpl_joints2d = smpl_joints2d.cpu().numpy()

            orig_height, orig_width = image_shape[:2]
            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=orig_width,
                img_height=orig_height
            )

            joints2d_img_coord = convert_crop_coords_to_orig_img(
                bbox=bboxes,
                keypoints=smpl_joints2d,
                crop_size=224,
            )

            output_dict = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'joints2d_img_coord': joints2d_img_coord,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

            vibe_results[person_id] = output_dict

        del model

        joblib.dump(vibe_results, os.path.join(output_folder, "vibe_output.pkl"))

        print(f'Results saved to \"{os.path.join(output_folder, "vibe_output.pkl")}\".')
        return vibe_results

    def _render(self, image_folder, image_shape, num_frames, vibe_results):
        wireframe = False
        sideview = False
        save_obj = False
        display = True
        output_path = '/tmp'  # No output
        # ========= Render results as a single video ========= #
        orig_height, orig_width = image_shape[:2]
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=wireframe)

        output_img_folder = f'{image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            if sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mc = mesh_color[person_id]

                mesh_filename = None

                if save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

                if sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        angle=270,
                        axis=[0, 1, 0],
                    )

            if sideview:
                img = np.concatenate([img, side_img], axis=1)

            # cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if display:
            cv2.destroyAllWindows()

    def _rm_tmp(self, image_folder):
        shutil.rmtree(image_folder)